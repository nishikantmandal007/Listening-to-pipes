import torch

# Load one of the tensors
pressure_tensor = torch.load("data/processed/Dynamic Pressure Sensor/p_leak.pt")
hydrophone_tensor = torch.load("data/processed/Hydrophone/h_leak.pt")

# Check their dimensions
print(pressure_tensor.shape)
print(hydrophone_tensor.shape)
