import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(x_concat)
        return self.sigmoid(spatial_att) * x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(current_channels, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(features[-1])
            features.append(new_feature)
        return torch.cat(features, dim=1)

class DenseNet(nn.Module):
    def __init__(self, in_channels, num_blocks=4, growth_rate=32):
        super(DenseNet, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dense_blocks = nn.ModuleList()
        
        current_channels = 64
        for _ in range(num_blocks):
            block = DenseBlock(current_channels, growth_rate)
            self.dense_blocks.append(block)
            current_channels += growth_rate * 4
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_channels, 1024)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.dense_blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class WaveNet(nn.Module):
    def __init__(self, in_channels):
        super(WaveNet, self).__init__()
        self.dilated_convs = nn.ModuleList()
        dilation_rates = [1, 2, 4, 8, 16]
        
        for dilation in dilation_rates:
            conv = nn.Conv1d(
                in_channels, 
                in_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation
            )
            self.dilated_convs.append(conv)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels * len(dilation_rates), 128)

    def forward(self, x):
        features = []
        for conv in self.dilated_convs:
            feat = conv(x)
            features.append(self.global_pool(feat).squeeze(-1))
        
        x = torch.cat(features, dim=1)
        return self.fc(x)

class MultimodalSensorClassifier(nn.Module):
    def __init__(self):
        super(MultimodalSensorClassifier, self).__init__()
        
        # CBAM Modules
        self.pressure_cbam = CBAM(in_channels=1)
        self.hydrophone_cbam = CBAM(in_channels=1)
        
        # Feature Extractors
        self.pressure_densenet = DenseNet(in_channels=1)
        self.hydrophone_wavenet = WaveNet(in_channels=1)
        
        # MLP Classifier
        self.mlp = nn.Sequential(
            nn.Linear(1152, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, pressure_data, hydrophone_data):
        # CBAM Processing
        pressure_cbam = self.pressure_cbam(pressure_data)
        hydrophone_cbam = self.hydrophone_cbam(hydrophone_data)
        
        # Feature Extraction
        pressure_features = self.pressure_densenet(pressure_cbam)
        hydrophone_features = self.hydrophone_wavenet(hydrophone_cbam)
        
        # Concatenate Features
        combined_features = torch.cat([pressure_features, hydrophone_features], dim=1)
        
        # Classification
        output = self.mlp(combined_features)
        
        return output

# Model Testing Function
def test_model():
    # Initialize the model
    model = MultimodalSensorClassifier()
    
    # Create dummy input tensors
    pressure_data = torch.rand(16, 1, 1000)
    hydrophone_data = torch.rand(16, 1, 1000)
    
    # Forward pass
    output = model(pressure_data, hydrophone_data)
    
    # Verify output shape
    print("Input Pressure Data Shape:", pressure_data.shape)
    print("Input Hydrophone Data Shape:", hydrophone_data.shape)
    print("Output Shape:", output.shape)
    assert output.shape == (16, 1), "Output shape does not match expected dimensions"
    print("Model forward pass successful!")

# Run the test if the script is executed directly
if __name__ == "__main__":
    test_model()