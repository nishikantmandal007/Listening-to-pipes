import os
import pandas as pd
import numpy as np
import torch
from pykrige.ok import OrdinaryKriging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocess_and_save(file_leak, file_noleak, output_dir='output/', batch_size=5000, resolution=25):
    os.makedirs(output_dir, exist_ok=True)

    def load_and_normalize(file):
        """
        Load data from CSV and normalize p1, p2.
        """
        data = pd.read_csv(file)
        scaler = MinMaxScaler()
        data[['p1', 'p2']] = scaler.fit_transform(data[['p1', 'p2']])
        return data

    def truncate_datasets(data1, data2):
        """
        Truncate datasets to have the same number of samples.
        """
        min_length = min(len(data1), len(data2))
        return data1[:min_length], data2[:min_length]

    def kriging_interpolation(data, resolution):
        """
        Applies Kriging interpolation to a smaller batch of data.
        """
        t = data['Sample'].values
        p1 = data['p1'].values
        p2 = data['p2'].values

        grid_t = np.linspace(t.min(), t.max(), resolution)

        kriging_p1 = OrdinaryKriging(t, np.zeros_like(t), p1, variogram_model='linear')
        interpolated_p1, _ = kriging_p1.execute('grid', grid_t, np.zeros_like(grid_t))

        kriging_p2 = OrdinaryKriging(t, np.zeros_like(t), p2, variogram_model='linear')
        interpolated_p2, _ = kriging_p2.execute('grid', grid_t, np.zeros_like(grid_t))

        interpolated = np.stack((interpolated_p1, interpolated_p2), axis=-1)
        return interpolated

    def process_in_batches(data, resolution, filename_prefix):
        """
        Process the data in batches and save intermediate tensors.
        """
        num_batches = len(data) // batch_size + int(len(data) % batch_size != 0)
        combined_tensor = []

        for i in range(num_batches):
            print(f"Processing batch {i + 1}/{num_batches}...")
            batch_data = data[i * batch_size: (i + 1) * batch_size]
            interpolated = kriging_interpolation(batch_data, resolution)
            batch_tensor = torch.tensor(interpolated, dtype=torch.float32)
            combined_tensor.append(batch_tensor)

        combined_tensor = torch.cat(combined_tensor, dim=0)
        torch.save(combined_tensor, os.path.join(output_dir, f'{filename_prefix}_tensor.pt'))
        return combined_tensor

    def visualize_and_save(leak_tensor, noleak_tensor, output_dir):
        """
        Visualize and save interpolated data.
        """
        os.makedirs(output_dir, exist_ok=True)

        diff_leak = leak_tensor[:, 1] - leak_tensor[:, 0]
        diff_noleak = noleak_tensor[:, 1] - noleak_tensor[:, 0]

        plt.figure(figsize=(14, 8))

        # Leak: p2 - p1
        plt.subplot(2, 3, 1)
        plt.plot(diff_leak.numpy(), label='Leak: p2-p1', color='red')
        plt.title('Leak: p2 - p1')
        plt.legend()

        # No-Leak: p2 - p1
        plt.subplot(2, 3, 2)
        plt.plot(diff_noleak.numpy(), label='No-Leak: p2-p1', color='blue')
        plt.title('No-Leak: p2 - p1')
        plt.legend()

        # Comparison of p1
        plt.subplot(2, 3, 3)
        plt.plot(leak_tensor[:, 0].numpy(), label='Leak: p1', color='orange')
        plt.plot(noleak_tensor[:, 0].numpy(), label='No-Leak: p1', color='cyan')
        plt.title('Comparison of p1')
        plt.legend()

        # Comparison of p2
        plt.subplot(2, 3, 4)
        plt.plot(leak_tensor[:, 1].numpy(), label='Leak: p2', color='green')
        plt.plot(noleak_tensor[:, 1].numpy(), label='No-Leak: p2', color='purple')
        plt.title('Comparison of p2')
        plt.legend()

        # Combined Diff
        plt.subplot(2, 3, 5)
        plt.plot(diff_leak.numpy(), label='Leak: Diff', color='red', linestyle='--')
        plt.plot(diff_noleak.numpy(), label='No-Leak: Diff', color='blue', linestyle='--')
        plt.title('Difference Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_plots.png'), dpi=300)
        plt.close()

    # Load and truncate datasets
    leak_data = load_and_normalize(file_leak)
    noleak_data = load_and_normalize(file_noleak)
    leak_data, noleak_data = truncate_datasets(leak_data, noleak_data)

    # Process and save tensors
    leak_tensor = process_in_batches(leak_data, resolution, 'leak')
    noleak_tensor = process_in_batches(noleak_data, resolution, 'noleak')

    # Visualize and save plots
    visualize_and_save(leak_tensor, noleak_tensor, output_dir)

    print(f"Tensors saved to {output_dir}")
    print(f"Plots saved as 'comparison_plots.png' in {output_dir}")

    return leak_tensor, noleak_tensor

# Paths to your files
leak_pth = "/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/raw/Dynamic Pressure Sensor/Leak/merged_leaked.csv"
noleak_pth = "/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/raw/Dynamic Pressure Sensor/No-Leak/merged_no-leak.csv"

leak_tensor, noleak_tensor = preprocess_and_save(leak_pth, noleak_pth)
