import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the audio file
file_path = 'C:\\Users\\Acer\\work\\git\\free-spoken-digit-dataset\\recordings\\9_yweweler_36.wav'

waveform, sr = torchaudio.load(file_path)

# 2. Create the mel spectrogram
n_fft = 2048
hop_length = 128
n_mels = 128

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    power=2.0
)

S = mel_spectrogram(waveform)

# Convert to dB scale
S_db = torchaudio.transforms.AmplitudeToDB()(S)

# 3. Apply Min/Max Normalization to the spectrogram
S_min = torch.min(S_db)
S_max = torch.max(S_db)
S_normalized = (S_db - S_min) / (S_max - S_min)

# 4. Plot the normalized mel spectrogram
plt.figure(figsize=(12, 8))
plt.imshow(S_normalized[0].numpy(), aspect='auto', origin='lower', interpolation='nearest')
plt.colorbar(format='%+2.0f dB')
plt.title('Normalized Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()

# 5. Denormalize the spectrogram before reconstruction
S_denormalized = S_normalized * (S_max - S_min) + S_min

# 6. Convert back from dB to power
S_reverse = torch.exp(S_denormalized * 0.11512925464970229)  # 10^(x/10) = e^(x * ln(10)/10)

# Now we can invert the mel spectrogram
inverse_mel_scale = torchaudio.transforms.InverseMelScale(
    n_stft=n_fft // 2 + 1,
    n_mels=n_mels,
    sample_rate=sr
)

griffin_lim = torchaudio.transforms.GriffinLim(
    n_fft=n_fft,
    hop_length=hop_length,
    n_iter=100
)

y_reverse = griffin_lim(inverse_mel_scale(S_reverse))

# 7. Enhance volume (optional)
volume_enhance_factor = 2.0  # Adjust this value to increase or decrease volume
y_reverse_enhanced = y_reverse * volume_enhance_factor

# Clip the enhanced audio to avoid distortion
y_reverse_enhanced = torch.clamp(y_reverse_enhanced, -1.0, 1.0)

# 8. Save the reconstructed audio
torchaudio.save('reconstructed_audio_spec_normalized.wav', y_reverse_enhanced, sample_rate=sr)

# 9. Plot the waveforms for comparison
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(waveform[0].numpy())
plt.title('Original Audio')
plt.subplot(2, 1, 2)
plt.plot(y_reverse_enhanced[0].numpy())
plt.title('Reconstructed Audio')
plt.tight_layout()
plt.show()