import torch.nn as nn

from .resnet import resnet50

CHANNELS = {
    "RN50" : 7,
    "ViT-L/14" : 768
}

class Classification(nn.Module):
    def __init__(self, num_classes=1):
        super(Classification, self).__init__()
        
        self.fc_layer = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, return_feature=False):
        x = x[3].view((x[3].shape)[0], -1)
        x = self.fc_layer(x)
        return x
