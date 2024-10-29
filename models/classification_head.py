import torch.nn as nn

from .resnet import resnet50

CHANNELS = {
    "RN50" : 7,
    "ViT-L/14" : 768
}

class Classification(nn.Module):
    def __init__(self, num_classes=1):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(2048 * 7 *7 , 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        # self.model = resnet50(pretrained=True)

    def forward(self, x, return_feature=False):
        print(x[3].shape)
        x = x[3].view(2, -1)
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc(x)
        return x
