import torch
import torch.nn as nn
import torchvision.models as models

class SliceLevelCNN(nn.Module):

    FEATURE_DIM = 2048
    NUM_SLICE_CLASSES = 6

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnext101_32x8d(weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.slice_classifier = nn.Linear(self.FEATURE_DIM, self.NUM_SLICE_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.slice_classifier(features)
        return features, logits