from PIL import Image
import torch.nn as nn
from models.resnet import resnet50
# from .FreqFusion import FreqFusion
from mmseg.models.decode_heads.ham_head import LightHamHeadFreqAware
from mmseg.ops.wrappers import resize

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}
# Reconstruction branch using ResNet50
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

        self.head = LightHamHeadFreqAware(
            in_channels=[256, 512], 
            channels=CHANNELS["RN50"],
            in_index=[0, 1],
            num_classes=3)
                
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.head(x)  # Output is (batch_size, 2048, 7, 7)
        features = resize(features, (224, 224))

        return features