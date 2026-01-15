import torch.nn as nn
from model.backbone import Backbone

class Detector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = Backbone()

        # Force feature map to fixed size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features)

        bbox = self.bbox_head(features)
        cls = self.class_head(features)

        return bbox, cls
