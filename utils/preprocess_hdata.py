import os
import numpy as np
import torch
import librosa
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_hydrophone_data(file_leak_h1, file_leak_h2, file_noleak_h1, file_noleak_h2, output_dir='output/', batch_size=5000, resolution=25):
    os.makedirs(output_dir, exist_ok=True)
    
    def load_and_trim(file, target_length):
        """
        Loads .wav file and trims it to the target length (in seconds).
        """
        y, sr = librosa.load(file, sr=None)
        trim_length = int(target_length * sr)
        y_trimmed = y[:trim_length]
        return y_trimmed, sr

    def normalize_audio(audio_data):
        """
        Normalize audio data to the range [0, 35.55].
        """
        scaler = MinMaxScaler(feature_range=(0, 35.55))
        audio_data = scaler.fit_transform(audio_data.reshape(-1, 1))
        return audio_data.flatten()

    def apply_wavelet_transform(audio_data, wavelet='db4', level=3):
        """
        Applies wavelet transform to audio data.
        """
        coeffs = pywt.wavedec(audio_data, wavelet, level=level)
        return np.concatenate(coeffs)

    def process_audio_files(file_h1, file_h2, target_length):
        """
        Processes two hydrophone audio files (H1 and H2).
        """
        audio_h1, sr = load_and_trim(file_h1, target_length)
        audio_h2, _ = load_and_trim(file_h2, target_length)

        # Normalize the audio data
        audio_h1 = normalize_audio(audio_h1)
        audio_h2 = normalize_audio(audio_h2)

        # Apply wavelet transform
        audio_h1_transformed = apply_wavelet_transform(audio_h1)
        audio_h2_transformed = apply_wavelet_transform(audio_h2)

        return audio_h1_transformed, audio_h2_transformed

    def process_in_batches(h1_transformed, h2_transformed, resolution, filename_prefix):
        """
        Process the transformed audio in batches and save intermediate tensors.
        """
        num_batches = len(h1_transformed) // batch_size + int(len(h1_transformed) % batch_size != 0)
        combined_tensor = []

        for i in range(num_batches):
            print(f"Processing batch {i + 1}/{num_batches}...")
            batch_h1 = h1_transformed[i * batch_size: (i + 1) * batch_size]
            batch_h2 = h2_transformed[i * batch_size: (i + 1) * batch_size]
            batch_tensor = torch.tensor(np.stack([batch_h1, batch_h2], axis=-1), dtype=torch.float32)
            combined_tensor.append(batch_tensor)

        combined_tensor = torch.cat(combined_tensor, dim=0)
        torch.save(combined_tensor, os.path.join(output_dir, f'{filename_prefix}_tensor.pt'))
        return combined_tensor

    def visualize_and_save(h1_leak, h2_leak, h1_noleak, h2_noleak, output_dir):
        """
        Visualizes and saves various plots for the processed hydrophone data.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Difference plots (h2 - h1)
        diff_leak = h2_leak - h1_leak
        diff_noleak = h2_noleak - h1_noleak
        
        plt.figure(figsize=(14, 8))

        # Leak: h2 - h1
        plt.subplot(2, 3, 1)
        plt.plot(diff_leak, label='Leak: h2 - h1', color='red')
        plt.title('Leak: h2 - h1')
        plt.legend()

        # No-Leak: h2 - h1
        plt.subplot(2, 3, 2)
        plt.plot(diff_noleak, label='No-Leak: h2 - h1', color='blue')
        plt.title('No-Leak: h2 - h1')
        plt.legend()

        # Combined Diff: Leak and No-Leak
        plt.subplot(2, 3, 3)
        plt.plot(diff_leak, label='Leak: Diff', color='red', linestyle='--')
        plt.plot(diff_noleak, label='No-Leak: Diff', color='blue', linestyle='--')
        plt.title('Comparison of Diff (Leak vs No-Leak)')
        plt.legend()

        # Visualizing the transformed audio (Wavelet)
        plt.subplot(2, 3, 4)
        plt.plot(h1_leak, label='Leak: H1', color='green')
        plt.plot(h2_leak, label='Leak: H2', color='orange')
        plt.title('Wavelet Transformed Leak Audio')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(h1_noleak, label='No-Leak: H1', color='purple')
        plt.plot(h2_noleak, label='No-Leak: H2', color='cyan')
        plt.title('Wavelet Transformed No-Leak Audio')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hydrophone_comparison_plots.png'), dpi=300)
        plt.close()

    # Process the hydrophone data
    target_length = 35.550625  # Target length in seconds (same as pressure data)
    
    # Process both leak and no-leak datasets
    h1_leak, h2_leak = process_audio_files(file_leak_h1, file_leak_h2, target_length)
    h1_noleak, h2_noleak = process_audio_files(file_noleak_h1, file_noleak_h2, target_length)
    
    # Process and save tensors for both leak and no-leak data
    leak_tensor = process_in_batches(h1_leak, h2_leak, resolution, 'leak')
    noleak_tensor = process_in_batches(h1_noleak, h2_noleak, resolution, 'noleak')

    # Visualize and save plots
    visualize_and_save(h1_leak, h2_leak, h1_noleak, h2_noleak, output_dir)

    print(f"Tensors saved to {output_dir}")
    print(f"Plots saved as 'hydrophone_comparison_plots.png' in {output_dir}")

    return leak_tensor, noleak_tensor

# Paths to your hydrophone .wav files
file_leak_h1 = "data/raw/Hydrophone/Leak/h1.wav"
file_leak_h2 = "data/raw/Hydrophone/Leak/h2.wav"
file_noleak_h1 = "data/raw/Hydrophone/No-Leak/h1.wav"
file_noleak_h2 = "data/raw/Hydrophone/No-Leak/h2.wav"

# Call the preprocessing function
leak_tensor, noleak_tensor = preprocess_hydrophone_data(file_leak_h1, file_leak_h2, file_noleak_h1, file_noleak_h2)
