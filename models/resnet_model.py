import torch.nn as nn

from .resnet import resnet50
from mmseg.models.necks.featurepyramid import Feature2Pyramid
from mmseg.models.backbones.resnet import ResNet

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class ResModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ResModel, self).__init__()

        # self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.fc = nn.Linear( CHANNELS[name], num_classes )
        
        # resnet = resnet50(pretrained=True)
        resnet = ResNet(depth=50, pretrained="open-mmlab://resnet50_v1c")
        # self.model = nn.Sequential(*list(resnet.children())[:-2])
        self.model = resnet
        # self.neck = Feature2Pyramid(embed_dim=CHANNELS["RN50"])

        # print(self.model)

    def forward(self, x, return_feature=False):
        features = self.model.forward(x)
        # features = self.neck.forward(features)
        return features