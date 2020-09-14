from torch import nn
from torchvision import models
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.resnet50(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        x = self.classifier(x)
        multilabel_proba = F.sigmoid(x)
        return multilabel_proba
