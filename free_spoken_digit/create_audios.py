import torch
import torchaudio
from arch.auto_encoder import VAE
from spectrogram_processor import SpectrogramProcessor
from common.seeding import set_seed
import pickle
import os
import numpy as np
import soundfile as sf
import configs


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
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms, replace=False)
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
        sf.write(save_path, signal,samplerate=sample_rate)
        

def save_signals_torch(signals, save_dir, sample_rate, file_paths = None):
    os.makedirs(save_dir, exist_ok=True)
    for i, signal in enumerate(signals):
        if file_paths is not None:
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(file_paths[i]))[0]+".wav")            
        else:
            save_path = os.path.join(save_dir, f"{i}.wav")
            
        
        torchaudio.save(save_path, signal.cpu(), sample_rate)       
                

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    set_seed(111)   
    
    # Processing and model configurations are shared with main.py via configs.py
    processing_config = configs.PROCESSING_CONFIGS[configs.SELECTION]
    processing_config.print_config()
    # Audio transforms are cheap; CPU also avoids CUDA STFT/NVRTC issues on some setups
    sp = SpectrogramProcessor(processing_config,device=torch.device('cpu'))
    
    SAVE_DIR_ORIGINAL = "free_spoken_digit/samples/original/"
    SAVE_DIR_GENERATED = "free_spoken_digit/samples/generated/"
    SAVE_DIR_SAMPLED = "free_spoken_digit/samples/sampled_from_prior/"
    SPECTROGRAMS_PATH = configs.SPECTROGRAMS_DIR
    MIN_MAX_VALUES_PATH = os.path.join(configs.SPECTROGRAMS_DIR, "min_max_values.pkl")

    model = VAE(**configs.build_model_kwargs()).to(device)
     
    # If you have a saved model, load it
    checkpoint = torch.load(configs.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode: no input noise, z = mu
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
    original_signals_recreated = sp.create_audio_from_spectrograms(sampled_specs, sampled_min_max_values)              
    
    # Generate audio from spectrograms using the model
    # Encode the original spectrograms, decode from the latent means,
    # then perform the reconstruction
    with torch.no_grad():    
        sampled_specs = torch.from_numpy(sampled_specs).to(device)
        # Add batch dimension if not present
        if sampled_specs.dim() == 3:
            sampled_specs = sampled_specs.unsqueeze(0)
        vae_generated_spectrograms, _, _, _ = model(sampled_specs)                
    vae_generated_spectrograms = vae_generated_spectrograms.cpu().numpy()
    vae_recreated_signals = sp.create_audio_from_spectrograms(vae_generated_spectrograms, sampled_min_max_values)      
    
    # Generate brand new sounds by decoding random latent vectors z ~ N(0, I).
    # Possible because spectrograms share one global dB range (no per-file min/max needed).
    num_prior_samples = 5
    with torch.no_grad():
        z = torch.randn(num_prior_samples, model.latent_dim, device=device)
        prior_spectrograms = model.decoder(z).cpu().numpy()
    global_min_max = sampled_min_max_values[0]
    prior_signals = sp.create_audio_from_spectrograms(prior_spectrograms, [global_min_max] * num_prior_samples)

    # Save audio signals
    sample_rate = processing_config.sample_rate
    save_signals_torch(vae_recreated_signals, SAVE_DIR_GENERATED, sample_rate, file_paths=sampled_file_paths)
    save_signals_torch(original_signals_recreated, SAVE_DIR_ORIGINAL, sample_rate, file_paths=sampled_file_paths)
    save_signals_torch(prior_signals, SAVE_DIR_SAMPLED, sample_rate)
    
    print("Completed!")
