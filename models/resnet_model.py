import torch.nn as nn

from .resnet import resnet50

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class ResModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ResModel, self).__init__()

        # self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.fc = nn.Linear( CHANNELS[name], num_classes )
        
        resnet = resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x, return_feature=False):
        features = self.model.forward(x)
        return features