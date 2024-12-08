import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_and_compare(p_leak_path, p_noleak_path, h_leak_path, h_noleak_path, output_dir='output/'):
    """
    Visualize and compare leak vs no-leak data with the difference between p2 - p1 for each.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tensors with weights_only=True to avoid security warnings
    p_leak_tensor = torch.load(p_leak_path, weights_only=True)
    p_noleak_tensor = torch.load(p_noleak_path, weights_only=True)
    h_leak_tensor = torch.load(h_leak_path, weights_only=True)
    h_noleak_tensor = torch.load(h_noleak_path, weights_only=True)

    # Align tensor sizes by trimming to the smaller size
    min_size = min(p_leak_tensor.shape[0], p_noleak_tensor.shape[0], h_leak_tensor.shape[0], h_noleak_tensor.shape[0])
    p_leak_tensor = p_leak_tensor[:min_size]
    p_noleak_tensor = p_noleak_tensor[:min_size]
    h_leak_tensor = h_leak_tensor[:min_size]
    h_noleak_tensor = h_noleak_tensor[:min_size]

    # Compute the differences: p2 - p1 for both leak and no-leak for both sensors
    p_leak_diff = p_leak_tensor[:, 1] - p_leak_tensor[:, 0]  # Leak: p2 - p1
    p_noleak_diff = p_noleak_tensor[:, 1] - p_noleak_tensor[:, 0]  # No-Leak: p2 - p1
    h_leak_diff = h_leak_tensor[:, 1] - h_leak_tensor[:, 0]  # Leak: p2 - p1
    h_noleak_diff = h_noleak_tensor[:, 1] - h_noleak_tensor[:, 0]  # No-Leak: p2 - p1

    # Set the color palette
    sns.set_palette("pastel")

    # Create plots for Dynamic Pressure Sensor
    plt.figure(figsize=(16, 12))

    # Plot 1: Difference between p2 and p1 for Leak vs No-Leak (Dynamic Pressure Sensor)
    plt.subplot(2, 2, 1)
    plt.plot(p_leak_diff.numpy(), label='', color='blue', alpha=0.7)
    plt.plot(p_noleak_diff.numpy(), label='No-Leak', color='green', alpha=0.7)
    plt.title('Dynamic Pressure Sensor: Difference in p2 - p1 (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Pressure Difference')
    plt.legend()

    # Plot 2: Distribution of Differences (Leak and No-Leak) (Dynamic Pressure Sensor)
    plt.subplot(2, 2, 2)
    sns.histplot(p_leak_diff.numpy(), kde=True, color='blue', label='Leak', bins=30, alpha=0.7)
    sns.histplot(p_noleak_diff.numpy(), kde=True, color='green', label='No-Leak', bins=30, alpha=0.7)
    plt.title('Dynamic Pressure Sensor: Distribution of p2 - p1 Differences (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Pressure Difference')
    plt.ylabel('Frequency')
    plt.legend()

    # Create plots for Hydrophone
    plt.figure(figsize=(16, 12))

    # Plot 3: Difference between p2 and p1 for Leak vs No-Leak (Hydrophone)
    plt.subplot(2, 2, 1)
    plt.plot(h_leak_diff.numpy(), label='Leak', color='blue', alpha=0.7)
    plt.plot(h_noleak_diff.numpy(), label='No-Leak', color='green', alpha=0.7)
    plt.title('Hydrophone: Difference in p2 - p1 (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Pressure Difference')
    plt.legend()

    # Plot 4: Distribution of Differences (Leak and No-Leak) (Hydrophone)
    plt.subplot(2, 2, 2)
    sns.histplot(h_leak_diff.numpy(), kde=True, color='blue', label='Leak', bins=30, alpha=0.7)
    sns.histplot(h_noleak_diff.numpy(), kde=True, color='green', label='No-Leak', bins=30, alpha=0.7)
    plt.title('Hydrophone: Distribution of p2 - p1 Differences (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Pressure Difference')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the comparison plots
    plt.tight_layout()
    p_comparison_plot_path = os.path.join(output_dir, 'dynamic_pressure_sensor_comparison.png')
    h_comparison_plot_path = os.path.join(output_dir, 'hydrophone_comparison.png')
    plt.savefig(p_comparison_plot_path, dpi=300)
    plt.close()

    plt.tight_layout()
    plt.savefig(h_comparison_plot_path, dpi=300)
    plt.close()

    print(f"Plots saved to {p_comparison_plot_path}")
    print(f"Plots saved to {h_comparison_plot_path}")

# Paths to the saved .pt files
p_leak_path = r'D:\IISF 2024\listening-to-pipes\data\processed\Dynamic Pressure Sensor\p_leak.pt'
p_noleak_path = r'D:\IISF 2024\listening-to-pipes\data\processed\Dynamic Pressure Sensor\p_noleak.pt'
h_leak_path = r'D:\IISF 2024\listening-to-pipes\data\processed\Hydrophone\h_leak.pt'
h_noleak_path = r'D:\IISF 2024\listening-to-pipes\data\processed\Hydrophone\h_noleak.pt'

# Generate comparison plots
visualize_and_compare(p_leak_path, p_noleak_path, h_leak_path, h_noleak_path)