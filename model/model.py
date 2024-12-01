import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel Attention
        y = self.channel_attention(x.transpose(1, 2)).transpose(1, 2)
        x = x * y

        # Spatial Attention
        y = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        y = self.spatial_attention(y)
        x = x * y
        return x


# DenseNet Feature Extractor for Pressure Data
class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, growth_rate=12, block_layers=3):
        super(DenseNetFeatureExtractor, self).__init__()
        self.layers = nn.ModuleList()
        channels = input_dim
        for _ in range(block_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(channels, growth_rate),
                    nn.ReLU(),
                    nn.BatchNorm1d(growth_rate),
                )
            )
            channels += growth_rate

    def forward(self, x):
        batch_size, time_steps, features = x.shape
        x = x.view(batch_size, -1)
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


# WaveNet Feature Extractor for Hydrophone Data
class WaveNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_layers=3, channels=16):
        super(WaveNetFeatureExtractor, self).__init__()

        # Dynamically set input channels based on input data
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)]  # The input channels come from input_dim
        )

        # Add additional convolution layers
        self.conv_layers.extend(
            [nn.Conv1d(channels, channels, kernel_size=3, padding=1) for _ in range(num_layers - 1)]
        )

        # Fully connected layer after the convolutional layers
        self.fc = nn.Linear(channels, 64)

    def forward(self, x):
        # Ensure the input tensor is in the correct shape (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # Convert to (batch_size, channels, sequence_length)

        # Apply convolution layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))  # Apply relu activation

        # Pooling over the sequence dimension (to aggregate the features)
        x = torch.mean(x, dim=2)
        
        # Pass the result through the fully connected layer
        x = self.fc(x)
        return x


# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# Full Leak Detection Model
class LeakDetectionModel(nn.Module):
    def __init__(self, pressure_dim, hydrophone_dim, output_dim=2):
        super(LeakDetectionModel, self).__init__()
        self.pressure_extractor = DenseNetFeatureExtractor(pressure_dim * 2)
        self.hydrophone_extractor = WaveNetFeatureExtractor(hydrophone_dim)
        self.attention = CBAM(channels=64)
        self.classifier = MLPClassifier(input_dim=128, output_dim=output_dim)

    def forward(self, pressure, hydrophone):

        # Concatenate pressure and hydrophone along the feature dimension
        combined_features = torch.cat((pressure, hydrophone), dim=-1)  # Shape: [batch_size, 50]
        
        # Add a channel dimension for Conv1d (conv1 expects [batch_size, channels, features])
        combined_features = combined_features.unsqueeze(1)  # Shape: [batch_size, 1, 50]

        # Pass through convolution and attention layers
        x = self.conv1(combined_features)  # Shape: [batch_size, 64, 50] after conv1
        x = F.relu(x)  # Apply ReLU activation
        x = self.attention(x)  # Apply attention mechanism (example)

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to shape [batch_size, 64*50]

        # Fully connected layers
        x = F.relu(self.fc1(x))  # Apply ReLU activation to fc1
        x = self.fc2(x)  # Output layer, no activation (softmax in loss function)

        return x
