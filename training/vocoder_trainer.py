import time
import logging
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.audio import MelSpectrogram, normalize_mel, stft_magnitude
from common.seeding import get_rng_state, set_rng_state
from training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)
STFT_RESOLUTIONS = ((64, 16), (128, 32), (256, 64), (512, 128))


def multi_resolution_stft_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Spectral convergence + log-magnitude L1, averaged over several STFT resolutions."""
    loss = 0.0
    for n_fft, hop in STFT_RESOLUTIONS:
        f, r = stft_magnitude(fake, n_fft, hop), stft_magnitude(real, n_fft, hop)
        loss = loss + torch.norm(r - f) / torch.norm(r) + F.l1_loss(torch.log(f), torch.log(r))
    return loss / len(STFT_RESOLUTIONS)


class VocoderTrainer:
    """GAN trainer for the vocoder: mel + multi-resolution STFT losses, adversarial and
    feature-matching losses after a warm-up, early stopping on validation mel loss, and exact resume."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module, mel: MelSpectrogram, top_db: float,
                 device: torch.device, lr: float = 5e-4, gan_start: int = 2000,
                 lambda_mel: float = 45.0, lambda_fm: float = 2.0):
        self.g, self.d, self.mel = generator.to(device), discriminator.to(device), mel.to(device)
        self.top_db, self.device, self.gan_start = top_db, device, gan_start
        self.lambda_mel, self.lambda_fm = lambda_mel, lambda_fm
        self.opt_g = torch.optim.AdamW(self.g.parameters(), lr=lr, betas=(0.8, 0.9))
        self.opt_d = torch.optim.AdamW(self.d.parameters(), lr=lr, betas=(0.8, 0.9))

    def log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Natural-log mel power, used for the mel reconstruction loss."""
        return torch.log(self.mel(wav).clamp(min=1e-5))

    def input_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Normalised mel the vocoder receives (same as the dataset spectrograms)."""
        return normalize_mel(self.mel(wav), self.top_db)

    def train_step(self, mel: torch.Tensor, wav: torch.Tensor, step: int) -> Dict[str, float]:
        """One generator (and, after warm-up, discriminator) update on aligned mel/waveform crops."""
        fake = self.g(mel)
        use_gan = step > self.gan_start
        if use_gan:
            d_loss = sum(F.relu(1 - real_logit).mean() + F.relu(1 + fake_logit).mean()
                         for (real_logit, _), (fake_logit, _) in zip(self.d(wav), self.d(fake.detach())))
            self.opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            self.opt_d.step()

        mel_loss = F.l1_loss(self.log_mel(fake), self.log_mel(wav))
        g_loss = self.lambda_mel * mel_loss + multi_resolution_stft_loss(fake, wav)
        if use_gan:
            fake_out = self.d(fake)
            with torch.no_grad():                      # real features are only targets for feature matching
                real_out = self.d(wav)
            adv = sum(F.relu(1 - logit).mean() for logit, _ in fake_out)
            fm = sum(F.l1_loss(f, r.detach()) for (_, ff), (_, rf) in zip(fake_out, real_out) for f, r in zip(ff, rf))
            g_loss = g_loss + adv + self.lambda_fm * fm
        self.opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        self.opt_g.step()
        return {'mel': mel_loss.item(), 'g': g_loss.item()}

    @torch.no_grad()
    def validate(self, mels: torch.Tensor, wavs: torch.Tensor, batch_size: int = 60) -> float:
        """Mean mel L1 between vocoded and real validation clips."""
        self.g.eval()
        losses = [F.l1_loss(self.log_mel(self.g(m)), self.log_mel(w)).item()
                  for m, w in zip(mels.split(batch_size), wavs.split(batch_size))]
        self.g.train()
        return sum(losses) / len(losses)

    def fit(self, train: tuple, val: tuple, steps: int, batch_size: int, crop_frames: int, eval_every: int,
            save_path: str, patience: int, log_every: int = 200, resume: bool = False) -> Dict[str, List[float]]:
        """Train on random aligned crops; validate on full clips; keep the best generator weights."""
        resume_path = str(Path(save_path).with_name(Path(save_path).stem + '_resume.pt'))
        history, stopper = {'step': [], 'train_mel': [], 'val_mel': []}, EarlyStopping(patience)
        first_step = self.load_resume(resume_path, history, stopper) + 1 if resume else 1
        running, start, step = [], time.time(), first_step - 1
        logger.info(f"Vocoder steps {first_step}..{steps} (batch {batch_size} × {crop_frames} frames, "
                    f"GAN from step {self.gan_start}, patience {patience})")
        try:
            for step in range(first_step, steps + 1):
                running.append(self.train_step(*self.random_crops(*train, batch_size, crop_frames), step)['mel'])
                if step % log_every == 0:
                    ms = (time.time() - start) / (step - first_step + 1) * 1000
                    logger.info(f"step {step:>6}/{steps} | mel {sum(running[-log_every:]) / log_every:.4f} | "
                                f"{ms:.0f} ms/step | ETA {(steps - step) * ms / 60000:.0f} min")
                if step % eval_every == 0:
                    stop = self._evaluate(step, running, val, history, stopper, save_path)
                    running = []
                    self.save_resume(resume_path, step, history, stopper)
                    if stop:
                        break
        except KeyboardInterrupt:
            self.save_resume(resume_path, step - 1, history, stopper)
            logger.info(f"Interrupted at step {step}; state saved to {resume_path}. Continue with --resume")
            raise SystemExit(130)
        logger.info(f"Finished in {(time.time() - start) / 60:.1f} min, best val mel {stopper.best:.4f}")
        return history

    def random_crops(self, mels: torch.Tensor, wavs: torch.Tensor, batch_size: int, crop_frames: int) -> tuple:
        """Aligned random crops: frames [f, f + L) <-> samples [f * hop, (f + L - 1) * hop)."""
        hop = self.g.hop_length
        idx = torch.randint(0, len(mels), (batch_size,), device=self.device)
        starts = torch.randint(0, mels.shape[-1] - crop_frames + 1, (batch_size,), device=self.device)
        frames = starts[:, None] + torch.arange(crop_frames, device=self.device)
        samples = starts[:, None] * hop + torch.arange((crop_frames - 1) * hop, device=self.device)
        return mels[idx[:, None], :, frames].transpose(1, 2), wavs[idx[:, None], samples]

    def _evaluate(self, step: int, running: List[float], val: tuple, history: Dict[str, List[float]],
                  stopper: EarlyStopping, save_path: str) -> bool:
        """Validate, save the generator if best, log; returns True when early stopping triggers."""
        val_mel = self.validate(*val)
        history['step'].append(step)
        history['train_mel'].append(sum(running) / max(len(running), 1))
        history['val_mel'].append(val_mel)
        improved = stopper.step(val_mel)
        if improved:
            torch.save({'step': step, 'history': history, 'model_state_dict': self.g.state_dict()}, save_path)
        saved = ' (saved)' if improved else ''
        logger.info(f"validation @ {step} | val mel {val_mel:.4f} | best {stopper.best:.4f}{saved} | "
                    f"patience {stopper.bad_checks}/{stopper.patience}")
        if stopper.should_stop:
            logger.info(f"Early stopping at step {step}: no improvement for {stopper.patience} validations")
        return stopper.should_stop

    def save_resume(self, path: str, step: int, history: Dict[str, List[float]], stopper: EarlyStopping) -> None:
        """Save everything needed to continue training exactly where it stopped."""
        torch.save({'step': step, 'g': self.g.state_dict(), 'd': self.d.state_dict(),
                    'opt_g': self.opt_g.state_dict(), 'opt_d': self.opt_d.state_dict(), 'history': history,
                    'early_stopping': stopper.state_dict(), 'rng_state': get_rng_state()}, path)

    def load_resume(self, path: str, history: Dict[str, List[float]], stopper: EarlyStopping) -> int:
        """Restore a resume checkpoint; returns the last completed step."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.g.load_state_dict(state['g'])
        self.d.load_state_dict(state['d'])
        self.opt_g.load_state_dict(state['opt_g'])
        self.opt_d.load_state_dict(state['opt_d'])
        history.update(state['history'])
        stopper.load_state_dict(state['early_stopping'])
        set_rng_state(state['rng_state'])
        logger.info(f"Resumed from {path} at step {state['step']} (best val mel {stopper.best:.4f})")
        return state['step']
