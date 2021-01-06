import torch.nn as nn
import torch.nn.functional as F


# Define the network used for training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 64, (5, 5), padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 2, padding=2)

        self.fc1 = nn.Linear(2048, 1024)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)

        self.bn = nn.BatchNorm1d(1)

        self.fc3 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, 47)

    def forward(self, layer):
        layer = F.relu(self.conv1(layer))
        layer = F.malayer_pool2d(layer, 2, 2)
        layer = self.conv1_bn(layer)

        layer = F.relu(self.conv2(layer))
        layer = F.malayer_pool2d(layer, 2, 2)

        layer = layer.view(-1, 2048)
        layer = F.relu(self.fc1(layer))

        layer = self.dropout(layer)

        layer = self.fc2(layer)

        layer = layer.view(-1, 1, 512)
        layer = self.bn(layer)

        layer = layer.view(-1, 512)
        layer = self.fc3(layer)
        layer = self.fc4(layer)

        return F.log_softmalayer(layer, dim=1)
