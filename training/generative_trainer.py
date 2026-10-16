import time
import logging
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from training.early_stopping import EarlyStopping
from common.seeding import get_rng_state, set_rng_state

logger = logging.getLogger(__name__)


class GenerativeTrainer:
    """Step-based trainer for DDPM / Flow Matching with EMA weights and mixed precision.

    Data is passed as GPU tensors in [-1, 1]; the whole dataset fits in memory, which avoids
    DataLoader overhead on a 3k-clip dataset.
    """

    def __init__(self, method: nn.Module, device: torch.device, lr: float = 2e-4,
                 warmup_steps: int = 1000, ema_decay: float = 0.9995, grad_clip: float = 1.0):
        self.method = method.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(method.parameters(), lr=lr, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))
        self.ema = AveragedModel(method, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        self.grad_clip = grad_clip

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """One optimisation step; returns the loss."""
        self.method.train()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            loss = self.method.loss(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.method.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update_parameters(self.method)
        return loss.item()

    @torch.no_grad()
    def validate(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, repeats: int = 4) -> float:
        """Mean EMA-model loss over the validation set (repeated to average out random timesteps)."""
        self.ema.module.eval()
        losses = [self.ema.module.loss(x[i:i + batch_size], y[i:i + batch_size]).item()
                  for _ in range(repeats) for i in range(0, len(x), batch_size)]
        return sum(losses) / len(losses)

    def fit(self, train: tuple, val: tuple, steps: int, batch_size: int, eval_every: int,
            save_path: str, patience: int, log_every: int = 500, resume: bool = False) -> Dict[str, List[float]]:
        """Train up to `steps` steps; validate every eval_every steps, keep the best EMA weights at save_path,
        write a resume checkpoint next to it and stop early after `patience` validations without improvement."""
        x_train, y_train = train
        resume_path = str(Path(save_path).with_name(Path(save_path).stem + '_resume.pt'))
        history = {'step': [], 'train_loss': [], 'val_loss': []}
        stopper = EarlyStopping(patience)
        first_step = self.load_resume(resume_path, history, stopper) + 1 if resume else 1
        running, start, step = [], time.time(), first_step - 1
        logger.info(f"Training steps {first_step}..{steps} (batch {batch_size}, {len(x_train)} clips, "
                    f"validate every {eval_every}, patience {patience})")
        try:
            for step in range(first_step, steps + 1):
                idx = torch.randint(0, len(x_train), (batch_size,), device=self.device)
                running.append(self.train_step(x_train[idx], y_train[idx]))
                if step % log_every == 0:
                    ms = (time.time() - start) / (step - first_step + 1) * 1000
                    logger.info(f"step {step:>6}/{steps} | loss {sum(running[-log_every:]) / log_every:.4f} | "
                                f"lr {self.scheduler.get_last_lr()[0]:.1e} | {ms:.0f} ms/step | "
                                f"ETA {(steps - step) * ms / 60000:.0f} min")
                if step % eval_every == 0:
                    stop = self._evaluate(step, running, val, batch_size, history, stopper, save_path)
                    running = []
                    self.save_resume(resume_path, step, history, stopper)
                    if stop:
                        break
        except KeyboardInterrupt:
            self.save_resume(resume_path, step - 1, history, stopper)
            logger.info(f"Interrupted at step {step}; state saved to {resume_path}. Continue with --resume")
            raise SystemExit(130)
        self.save_history(save_path, history)
        logger.info(f"Finished in {(time.time() - start) / 60:.1f} min, best val (EMA) {stopper.best:.4f}")
        return history

    def _evaluate(self, step: int, running: List[float], val: tuple, batch_size: int,
                  history: Dict[str, List[float]], stopper: EarlyStopping, save_path: str) -> bool:
        """Validate, save the EMA weights if best, log; returns True when early stopping triggers."""
        val_loss = self.validate(*val, batch_size)
        history['step'].append(step)
        history['train_loss'].append(sum(running) / max(len(running), 1))
        history['val_loss'].append(val_loss)
        improved = stopper.step(val_loss)
        if improved:
            self.save(save_path, step, history)
        saved = ' (saved)' if improved else ''
        logger.info(f"validation @ {step} | val (EMA) {val_loss:.4f} | best {stopper.best:.4f}{saved} | "
                    f"patience {stopper.bad_checks}/{stopper.patience}")
        if stopper.should_stop:
            logger.info(f"Early stopping at step {step}: no improvement for {stopper.patience} validations")
        return stopper.should_stop

    def save_resume(self, path: str, step: int, history: Dict[str, List[float]], stopper: EarlyStopping) -> None:
        """Save everything needed to continue training exactly where it stopped."""
        torch.save({'step': step, 'method': self.method.state_dict(), 'ema': self.ema.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                    'scaler': self.scaler.state_dict(), 'history': history,
                    'early_stopping': stopper.state_dict(), 'rng_state': get_rng_state()}, path)

    def load_resume(self, path: str, history: Dict[str, List[float]], stopper: EarlyStopping) -> int:
        """Restore a resume checkpoint; returns the last completed step."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.method.load_state_dict(state['method'])
        self.ema.load_state_dict(state['ema'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        history.update(state['history'])
        stopper.load_state_dict(state['early_stopping'])
        set_rng_state(state['rng_state'])
        logger.info(f"Resumed from {path} at step {state['step']} (best val {stopper.best:.4f})")
        return state['step']

    def save(self, path: str, step: int, history: Dict[str, List[float]]) -> None:
        """Save EMA weights of the denoising network (fp16 to keep checkpoints small)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        weights = {k: v.half() if v.is_floating_point() else v
                   for k, v in self.ema.module.model.state_dict().items()}
        torch.save({'step': step, 'history': history, 'model_state_dict': weights}, path)

    @staticmethod
    def save_history(path: str, history: Dict[str, List[float]]) -> None:
        """Store the full training history in the (best) checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint['history'] = history
        torch.save(checkpoint, path)
