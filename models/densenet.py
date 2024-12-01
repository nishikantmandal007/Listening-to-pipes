import torch
import torch.nn as nn

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