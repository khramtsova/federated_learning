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


class CNNMnist(nn.Module):
    def __init__(self, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(LinearModel, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        #self.bn2 =nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        #x = self.fc3(x)
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
