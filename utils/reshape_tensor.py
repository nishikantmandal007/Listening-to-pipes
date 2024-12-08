import torch
import torch.nn.functional as F

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

# Now both tensors should have the same first and second dimensions
print(pressure_tensor.shape)  # Should be [4575, 25, 2]
print(hydrophone_tensor.shape)  # Should be [4575, 25, 2]

# Example concatenation along the last dimension
combined_tensor = torch.cat((pressure_tensor, hydrophone_tensor), dim=-1)
print(combined_tensor.shape)  # Should be [4575, 25, 4]