import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        # W_out = (W_in - F +2P) / S
        # F-filter, P-padding, S-stride
        # 3*32*32

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        # 6*28*28
        self.pool = nn.MaxPool2d(2, 2)
        # 6*14*14

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 16*10*10

        self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
