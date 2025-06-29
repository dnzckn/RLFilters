
import torch
import torch.nn as nn
import torch.nn.functional as F

class EM_Emulator(nn.Module):
    """Convolutional Neural Network for emulating EM properties."""
    def __init__(self, grid_size, num_freq_points):
        super(EM_Emulator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size of the flattened features after conv and pooling layers
        feature_size = 64 * (grid_size // 2 // 2 // 2) * (grid_size // 2 // 2 // 2)

        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, num_freq_points)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
