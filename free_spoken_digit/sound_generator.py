import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import pickle
from common.seeding import set_seed
from arch.auto_encoder import VAE
import librosa
class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to a tensor."""
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
        
    def normalise(self, array):
        if not isinstance(array, torch.Tensor):
            array = torch.from_numpy(array)
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
        
    def denormalise(self, norm_array, original_min, original_max):
        if not isinstance(norm_array, torch.Tensor):
            norm_array = torch.from_numpy(norm_array)
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

class SoundGenerator:
    """SoundGenerator is responsible for generating audios from spectrograms using PyTorch."""
    
    def __init__(self, vae, hop_length, n_fft=2048):
        self.vae = vae
        self.hop_length = hop_length
        self.n_fft = n_fft
        self._min_max_normaliser = MinMaxNormaliser(0, 1)
        self.device = next(vae.parameters()).device
        
    def generate(self, spectrograms, min_max_values):
        # Convert to tensor if needed
        if not isinstance(spectrograms, torch.Tensor):
            spectrograms = torch.from_numpy(spectrograms).float()
        
        # Move to correct device
        spectrograms = spectrograms.to(self.device)
        
        # Inference mode
        with torch.no_grad():
            # Get reconstructions from VAE
            reconstructions, _, _, _ = self.vae(spectrograms)
            
            # Convert reconstructions to audio
            signals = self.convert_spectrograms_to_audio(
                reconstructions, 
                min_max_values
            )
            
            
        return signals
    
    def db_to_amplitude(self, db_spec):
        """Convert dB spectrogram to amplitude spectrogram"""
        return torch.pow(10.0, db_spec / 20.0)
    
    def griffin_lim(self, mel_spectrograms, n_iter=100):
        device = mel_spectrograms.device
        
        mel_basis = librosa.filters.mel(
            sr=22050,
            n_fft=self.n_fft,
            n_mels=64
        )
        mel_inverse = np.linalg.pinv(mel_basis)
        
        # Convert mel to linear spectrogram
        linear_spec = torch.tensor(
            mel_inverse @ mel_spectrograms.cpu().numpy(), 
            dtype=torch.complex64,
            device=device
        )
        
        # Initialize phase with better estimate
        angles = torch.exp(1j * torch.angle(linear_spec + 1e-9))
        
        for _ in range(n_iter):
            full_spec = linear_spec * angles
            inverse = torch.istft(
                full_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalized=True,
                return_complex=False
            )
            rebuilt = torch.stft(
                inverse,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalized=True,
                return_complex=True
            )
            angles = rebuilt / (torch.abs(rebuilt) + 1e-8)
        
        full_spec = linear_spec * angles
        signal = torch.istft(
            full_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            return_complex=False
        )
              
        return signal
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        
        if not isinstance(spectrograms, torch.Tensor):
            spectrograms = torch.from_numpy(spectrograms)
        
        spectrograms = spectrograms.to(self.device)
        
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # Get spectrogram and reshape
            log_spectrogram = spectrogram[:, :, 0]
            
            # Denormalize using min/max values that were saved AFTER normalization
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, 
                min_max_value["min"], 
                min_max_value["max"]
            )
            
            # Convert from dB to amplitude (with power compensation)
            spec = self.db_to_amplitude(denorm_log_spec)
            
            # Apply Griffin-Lim
            signal = self.griffin_lim(spec)
            signal = signal.cpu().numpy()
            signals.append(signal)
        
        return signals

def load_fsdd(spectrograms_path):
   x_train = []
   file_paths = []
   for root, _, file_names in os.walk(spectrograms_path):
       for file_name in file_names:
           if file_name.endswith('.npy'):
               file_path = os.path.join(root, file_name)
               spectrogram = np.load(file_path)
               spectrogram = np.transpose(spectrogram, (1, 2, 0))  # Reorder dimensions
               x_train.append(spectrogram)
               file_paths.append(file_path)
   x_train = np.array(x_train) # -> (3000, 64, 64, 1) # default. This will vary based on how you create spectrograms
   return x_train, file_paths



def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    sampled_file_paths = [file_paths[index] for index in sampled_indexes]
    
    # Normalize paths for comparison
    normalized_min_max_paths = {os.path.normpath(k): v for k, v in min_max_values.items()}
    sampled_min_max_values = [normalized_min_max_paths[os.path.normpath(fp)] for fp in sampled_file_paths]
    
    return sampled_spectrograms, sampled_min_max_values

def save_signals(signals, save_dir, sample_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, f"{i}.wav")
        sf.write(save_path, signal, sample_rate)

if __name__ == "__main__":
    # Constants
    N_FFT=512  # Make sure you have this correct when spectrograms were created
    HOP_LENGTH = 256 # Make sure you have this correct when spectrograms were created
    SAVE_DIR_ORIGINAL = "/samples/original/"
    SAVE_DIR_GENERATED = "/samples/generated/"
    SPECTROGRAMS_PATH = "C:/Users/Acer/work/data/free-audio-spectrogram"
    MIN_MAX_VALUES_PATH = "C:/Users/Acer/work/data/free-audio-spectrogram/min_max_values.pkl"

     # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    set_seed(111)   
    
    # Initialize the model - make sure it has same structure as saved model
    model = VAE(
        input_shape=(64, 64, 1),
        conv_filters=(64, 32,16),
        conv_kernels=(3, 3, 3),
        conv_strides=(1, 2, 2),
        latent_dim=16,
        dropout_rate=0.1,
        noise_factor=0.2,
        seed=42
    ).to(device)
    
    # If you have a saved model, load it
    checkpoint = torch.load('free_spoken_digit/checkpoints/vae_free_spoken_digit_20241114_152708_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()
    
    # Initialize sound generator
    sound_generator = SoundGenerator(model,HOP_LENGTH, N_FFT)

    # Load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)
    
    # Sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(
        specs,
        file_paths,
        min_max_values,
        5
    )

    # Generate audio for sampled spectrograms
    signals = sound_generator.generate(sampled_specs, sampled_min_max_values)
    
    # Convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, 
        sampled_min_max_values
    )

    # Save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)