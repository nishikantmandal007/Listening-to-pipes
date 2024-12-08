import torch
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, pressure_tensor, hydrophone_tensor):
        self.pressure_tensor = pressure_tensor
        self.hydrophone_tensor = hydrophone_tensor

        # Ensure the tensors have the same shape
        assert self.pressure_tensor.shape == self.hydrophone_tensor.shape, "Tensors must have the same shape"

        # Combine the tensors along the last dimension
        self.combined_tensor = torch.cat((self.pressure_tensor, self.hydrophone_tensor), dim=-1)

    def __len__(self):
        return self.combined_tensor.size(0)

    def __getitem__(self, idx):
        return self.combined_tensor[idx]

# Load the tensors
pressure_tensor = torch.load("D:/IISF 2024/listening-to-pipes/data/processed/Dynamic Pressure Sensor/p_leak.pt")
hydrophone_tensor = torch.load("D:/IISF 2024/listening-to-pipes/data/processed/Hydrophone/h_leak.pt")

# Reshape hydrophone_tensor to have the same number of dimensions as pressure_tensor
hydrophone_tensor = hydrophone_tensor.view(-1, 1, 2)  # Shape: [284424, 1, 2]

# Determine the target length (assuming we want to match the first dimension of pressure_tensor)
target_length = pressure_tensor.size(0)

# Pad or truncate hydrophone_tensor to match the target length
if hydrophone_tensor.size(0) > target_length:
    hydrophone_tensor = hydrophone_tensor[:target_length]
elif hydrophone_tensor.size(0) < target_length:
    padding = torch.zeros(target_length - hydrophone_tensor.size(0), 1, 2)
    hydrophone_tensor = torch.cat((hydrophone_tensor, padding), dim=0)

# Expand hydrophone_tensor to match the second dimension of pressure_tensor
hydrophone_tensor = hydrophone_tensor.expand(-1, pressure_tensor.size(1), -1)

# Create the dataset
dataset = CombinedDataset(pressure_tensor, hydrophone_tensor)