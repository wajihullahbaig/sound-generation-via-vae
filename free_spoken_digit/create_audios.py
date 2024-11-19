import torch
import torchaudio
from arch.auto_encoder import VAE
from spectrogram_processor import SpectrogramProcessor, ProcessingConfig
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
               x_train.append(spectrogram)
               file_paths.append(file_path)
   x_train = np.array(x_train) # -> (3000, 64, 64, 1) # default. This will vary based on how you create spectrograms
   return x_train, file_paths

def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    sampled_file_paths = [os.path.normpath(file_paths[index]) for index in sampled_indexes]
    
    # Normalize paths for comparison
    normalized_min_max_paths = {os.path.normpath(k): v for k, v in min_max_values.items()}
    sampled_min_max_values = [normalized_min_max_paths[os.path.normpath(fp)] for fp in sampled_file_paths]
    
    return sampled_spectrograms, sampled_min_max_values, sampled_file_paths

def save_signals_sf(signals, save_dir, sample_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, f"{i}.wav")
        sf.write(save_path, signal,samplerate=22050)
        

def save_signals_torch(signals, save_dir, sample_rate=22050, file_paths = None):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        if file_paths is not None:
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(file_paths[i]))[0]+".wav")            
        else:
            save_path = os.path.join(save_dir, f"{i}.wav")
            
        
        torchaudio.save(save_path, signal, sample_rate)       
                

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    set_seed(111)   
    
    # Configure preprocessing - Ensure you have correct configurations from spectrogram creation
    processing_configurations = {
    "256x64": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=512,        
        hop_length=64,     
        n_mels=64,         
        ),
    "128x128": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=2048,        
        hop_length=128,     
        n_mels=128,         
        ),    
    "32x32": ProcessingConfig(
        sample_rate=22050,
        duration=0.7425,      
        n_fft=2048,        
        hop_length=512,     
        n_mels=32,         
        )
    }
    selection = "128x128"
    processing_configurations[selection].print_config()
    sp = SpectrogramProcessor(processing_configurations[selection],device=device)
    
    SAVE_DIR_ORIGINAL = "free_spoken_digit/samples/original/"
    SAVE_DIR_GENERATED = "free_spoken_digit/samples/generated/"
    SPECTROGRAMS_PATH = "C:/Users/Acer/work/data/free-audio-spectrogram"
    MIN_MAX_VALUES_PATH = "C:/Users/Acer/work/data/free-audio-spectrogram/min_max_values.pkl"

    # Create model - set noise_factor > 0.0 to make VAE a DVAE
    # Note - The VAE input shape depends on the shape of spectrograms
    H = processing_configurations[selection].n_mels
    W = processing_configurations[selection].time_bins
    model = VAE(
        input_shape=(H, W,1),
        conv_filters=(64,32,16),
        conv_kernels=(3, 3, 3),
        conv_strides=(1, 2, 2),
        latent_dim=16,
        dropout_rate=0.2,
        noise_factor=0.1,
        seed=42
    ).to(device)
    
    # If you have a saved model, load it
    checkpoint = torch.load('free_spoken_digit/checkpoints/vae_free_spoken_digit_20241118_171519_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()
    
    # Load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)
    
    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)
    
    # Sample spectrograms + min max values
    sampled_specs, sampled_min_max_values, sampled_file_paths = select_spectrograms(
        specs,
        file_paths,
        min_max_values,
        5
    )

    # Generate audio from original spectrograms
    originaL_signals_recreated = sp.create_audio_from_spectrograms(sampled_specs, sampled_min_max_values)              
    
    # Generate audio from spectrograms using the model
    # Use the original audios to sample the latent space and create spectrograms
    # Then perform the reconsturction
    sampled_specs = torch.from_numpy(sampled_specs).to(device)
    model.eval()
    with torch.no_grad():    
        # Add batch dimension if not present
        if sampled_specs.dim() == 3:
            sampled_specs = sampled_specs.unsqueeze(0)
    vae_generated_spectrograms, _, _, _ = model(sampled_specs)                
    vae_generated_spectrograms = vae_generated_spectrograms.detach().cpu().numpy()
    vae_recreated_signals = sp.create_audio_from_spectrograms(vae_generated_spectrograms, sampled_min_max_values)      
    

    # Save audio signals
    save_signals_torch(vae_recreated_signals, SAVE_DIR_GENERATED, file_paths=sampled_file_paths)
    save_signals_torch(originaL_signals_recreated, SAVE_DIR_ORIGINAL, file_paths=sampled_file_paths)
    
    print("Completed!")