import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_and_compare(leak_tensor_path, noleak_tensor_path, output_dir='output/'):
    """
    Visualize and compare leak vs no-leak data with the difference between p2 - p1 for each.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load tensors with weights_only=True to avoid security warnings
    leak_tensor = torch.load(leak_tensor_path, weights_only=True)
    noleak_tensor = torch.load(noleak_tensor_path, weights_only=True)

    # Align tensor sizes by trimming to the smaller size
    min_size = min(leak_tensor.shape[0], noleak_tensor.shape[0])
    leak_tensor = leak_tensor[:min_size]
    noleak_tensor = noleak_tensor[:min_size]

    # Compute the differences: p2 - p1 for both leak and no-leak
    leak_diff = leak_tensor[:, 1] - leak_tensor[:, 0]  # Leak: p2 - p1
    noleak_diff = noleak_tensor[:, 1] - noleak_tensor[:, 0]  # No-Leak: p2 - p1

    # Fuse data: Combine leak and no-leak data for comparison
    combined_data = np.concatenate([leak_diff.numpy(), noleak_diff.numpy()])
    labels = ['Leak'] * len(leak_diff) + ['No-Leak'] * len(noleak_diff)

    # Ensure both combined_data and labels are 1-dimensional
    combined_data = combined_data.flatten()  # Flatten to 1D
    labels = np.array(labels).flatten()  # Flatten to 1D

    # Check if lengths match
    assert len(combined_data) == len(labels), f"Length mismatch: {len(combined_data)} != {len(labels)}"

    # Set the color palette
    sns.set_palette("pastel")

    # Create plots
    plt.figure(figsize=(16, 12))

    # Plot 1: Difference between p2 and p1 for Leak vs No-Leak
    plt.subplot(2, 2, 1)
    plt.plot(leak_diff.numpy(), label='Leak p2 - p1', color='blue', alpha=0.7)
    plt.plot(noleak_diff.numpy(), label='No-Leak p2 - p1', color='green', alpha=0.7)
    plt.title('Difference in p2 - p1 (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Pressure Difference')
    plt.legend()

    # Plot 2: Distribution of Differences (Leak and No-Leak)
    plt.subplot(2, 2, 2)
    sns.histplot(leak_diff.numpy(), kde=True, color='blue', label='Leak p2 - p1', bins=30, alpha=0.7)
    sns.histplot(noleak_diff.numpy(), kde=True, color='green', label='No-Leak p2 - p1', bins=30, alpha=0.7)
    plt.title('Distribution of p2 - p1 Differences (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Pressure Difference')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 3: Combined Data with Conditions (Boxplot)
    plt.subplot(2, 2, 3)
    sns.boxplot(x=labels, y=combined_data, palette="Set2")
    plt.title('Boxplot of p2 - p1 Differences (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Condition')
    plt.ylabel('Pressure Difference')

    # Plot 4: Density Plot for Leak and No-Leak Differences
    plt.subplot(2, 2, 4)
    sns.kdeplot(leak_diff.numpy(), shade=True, color='blue', label='Leak p2 - p1', alpha=0.7)
    sns.kdeplot(noleak_diff.numpy(), shade=True, color='green', label='No-Leak p2 - p1', alpha=0.7)
    plt.title('Density Plot of p2 - p1 Differences (Leak vs No-Leak)', fontsize=14)
    plt.xlabel('Pressure Difference')
    plt.ylabel('Density')
    plt.legend()

    # Save the comparison plot
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'leak_vs_noleak_difference_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()

    print(f"Plots saved to {comparison_plot_path}")

# Paths to the saved .pt files
leak_tensor_path = 'utils/output/final_leak_tensor.pt'
noleak_tensor_path = 'utils/output/final_noleak_tensor.pt'

# Generate comparison plots
visualize_and_compare(leak_tensor_path, noleak_tensor_path)
