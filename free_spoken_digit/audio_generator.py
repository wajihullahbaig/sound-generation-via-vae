from sound_generator import SoundGenerator
from arch.auto_encoder import VAE
import torch

class AudioGenerator:
    """Helper class for generating audio using the VAE"""
    def __init__(self, vae: VAE, sound_generator: SoundGenerator):
        self.vae = vae
        self.sound_generator = sound_generator
        self.device = next(vae.parameters()).device

    def generate_random_samples(self, num_samples: int = 1, temp: float = 1.0):
        """Generate random audio samples"""
        specs = self.vae.sample(num_samples, temp)
        # Create dummy min_max_values for the sound generator
        min_max_values = [{"min": -80, "max": 0} for _ in range(num_samples)]
        signals, _ = self.sound_generator.convert_spectrograms_to_audio(specs, min_max_values)
        return signals, specs

    def generate_interpolation(self, audio1, audio2, steps: int = 10):
        """Generate interpolation between two audio samples"""
        # Convert audio to spectrograms if needed
        spec1 = self.sound_generator.convert_audio_to_spectrogram(audio1)
        spec2 = self.sound_generator.convert_audio_to_spectrogram(audio2)
        
        # Interpolate
        specs = self.vae.interpolate(spec1, spec2, steps)
        
        # Convert back to audio
        min_max_values = [{"min": -80, "max": 0} for _ in range(steps)]
        signals, _ = self.sound_generator.convert_spectrograms_to_audio(specs, min_max_values)
        return signals, specs

    def generate_variations(self, audio, num_variations: int = 5, temp: float = 0.5):
        """Generate variations of an existing audio sample"""
        # Convert audio to spectrogram
        spec = self.sound_generator.convert_audio_to_spectrogram(audio)
        
        # Encode to get latent representation
        with torch.no_grad():
            mu, log_var = self.vae.encoder(spec.to(self.device))
            
        # Sample variations around the mean
        variations = []
        for _ in range(num_variations):
            z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) * temp
            variation = self.vae.decoder(z)
            variations.append(variation)
            
        specs = torch.stack(variations)
        min_max_values = [{"min": -80, "max": 0} for _ in range(num_variations)]
        signals, _ = self.sound_generator.convert_spectrograms_to_audio(specs, min_max_values)
        return signals, specs


if __name__ == "__main__":
    # Initialize models
    vae = VAE()
    sound_generator = SoundGenerator(vae, hop_length=256)
    audio_gen = AudioGenerator(vae, sound_generator)

    # Generate random samples
    samples, specs = audio_gen.generate_random_samples(
        num_samples=3, 
        temp=0.8
    )

    # Save the generated audio
    for i, signal in enumerate(samples):
        sf.write(f"generated_sample_{i}.wav", signal, 22050)

    # Load two audio files and generate interpolation
    audio1, sr = sf.read("sample1.wav")
    audio2, sr = sf.read("sample2.wav")
    
    interpolated, specs = audio_gen.generate_interpolation(
        audio1, 
        audio2, 
        steps=10
    )

    # Generate variations of an existing sample
    variations, specs = audio_gen.generate_variations(
        audio1,
        num_variations=5,
        temp=0.5
    )