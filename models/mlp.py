import torch
import torch.nn as nn
from .cbam import CBAM
from .densenet import DenseNet
from .wavenet import WaveNet
from .physics_loss import create_physics_informed_loss

class MultimodalPhysicsInformedClassifier(nn.Module):
    def __init__(self):
        super(MultimodalPhysicsInformedClassifier, self).__init__()
        
        # CBAM Modules for attention
        self.pressure_cbam = CBAM(in_channels=1)
        self.hydrophone_cbam = CBAM(in_channels=1)
        
        # Feature Extractors
        self.pressure_densenet = DenseNet(in_channels=1)
        self.hydrophone_wavenet = WaveNet(in_channels=1)
        
        # Simple MLP for classification
        self.mlp = nn.Sequential(
            # First layer - feature reduction
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer for binary classification
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, pressure_data, hydrophone_data):
        # Process pressure data
        pressure_cbam = self.pressure_cbam(pressure_data)
        pressure_features = self.pressure_densenet(pressure_cbam)
        
        # Process hydrophone data
        hydrophone_cbam = self.hydrophone_cbam(hydrophone_data)
        hydrophone_features = self.hydrophone_wavenet(hydrophone_cbam)
        
        # Combine features
        combined_features = torch.cat([pressure_features, hydrophone_features], dim=1)
        
        # Binary classification through MLP
        output = self.mlp(combined_features)
        
        return output

def create_physics_informed_model():
    """
    Factory function to create the physics-informed model
    
    Returns:
    - Configured MultimodalPhysicsInformedClassifier
    - Physics-informed loss function
    """
    model = MultimodalPhysicsInformedClassifier()
    loss_fn = create_physics_informed_loss()
    
    return model, loss_fn

# Example usage and testing
def test_model():
    model, loss_fn = create_physics_informed_model()
    
    # Create dummy input tensors
    pressure_data = torch.rand(16, 1, 1000)  # Batch size 16, 1 channel, 1000 time steps
    hydrophone_data = torch.rand(16, 1, 1000)
    targets = torch.rand(16, 1)  # Binary targets
    
    # Forward pass
    output = model(pressure_data, hydrophone_data)
    
    # Compute loss
    loss = loss_fn(output, targets, pressure_data, hydrophone_data)
    
    # Verify output shape
    print("Input Pressure Data Shape:", pressure_data.shape)
    print("Input Hydrophone Data Shape:", hydrophone_data.shape)
    print("Output Shape:", output.shape)
    print("Loss:", loss.item())
    
    assert output.shape == (16, 1), "Output shape does not match expected dimensions"
    print("Model forward pass and loss computation successful!")

if __name__ == "__main__":
    test_model()