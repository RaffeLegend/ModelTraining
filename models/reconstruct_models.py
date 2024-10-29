from PIL import Image
import torch.nn as nn
from models.resnet import resnet50
# from .FreqFusion import FreqFusion
from mmseg.models.decode_heads.ham_head import LightHamHead
from mmseg.ops.wrappers import resize

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}
# Reconstruction branch using ResNet50
class ReconstructionModel(nn.Module):
    def __init__(self):
        super(ReconstructionModel, self).__init__()
        
        # Load pre-trained ResNet50
        head = LightHamHead(in_channels=CHANNELS["RN50"], channels=3)
        self.head = nn.Sequential(*list(head.children())[:-2])
        
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.backbone(x)  # Output is (batch_size, 2048, 7, 7)
        
        # Reconstruct image
        reconstructed_image = self.upsample(features)
        return reconstructed_image
    
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

        self.head = LightHamHead(
            in_channels=[256, 512], 
            channels=CHANNELS["RN50"],
            in_index=[0, 1],
            num_classes=3)
                
        self.upsample = resize()
    def forward(self, x):
        # Extract features using ResNet50 backbone
        features = self.head(x)  # Output is (batch_size, 2048, 7, 7)
        features = self.upsample(features, (224, 224))

        return features