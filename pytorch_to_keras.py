from pytorch2keras.converter import pytorch_to_keras

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

import sys

class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
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
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }

path_to_weights = sys.argv[1]# "/home/jcmaas/Downloads/best.pth"
output_path = sys.argv[2] # "/home/jcmaas/Downloads/best_keras.nn"

var = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(
    0, 1, (64, 3, 7, 7)
)))

pytorch_model = ResNet("resnet50")
pytorch_model.load_state_dict(torch.load(path_to_weights)["model_state_dict"])

print(pytorch_model)

keras_model = pytorch_to_keras(pytorch_model, var)

keras_model.save(output_path)
