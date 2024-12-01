import torch
import torch.nn as nn

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