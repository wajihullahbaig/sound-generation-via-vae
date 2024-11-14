import torch
from arch.auto_encoder import VAE
from sound_generator import SoundGenerator
from common.seeding import set_seed
import pickle
import os
import numpy as np
import soundfile as sf


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
        sf.write(save_path, signal,samplerate=22050)

if __name__ == "__main__":
    # Constants
    N_FFT=512  # Make sure you have this correct when spectrograms were created
    HOP_LENGTH = 256 # Make sure you have this correct when spectrograms were created
    SAVE_DIR_ORIGINAL = "free_spoken_digit/samples/original/"
    SAVE_DIR_GENERATED = "free_spoken_digit/samples/generated/"
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
    
    print("Completed!")