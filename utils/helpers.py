import torch
from torch import nn
from torchvision import models


def load_model(path):
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as err:
        print(err)
        return None
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
    model.parameters = checkpoint['parameters']
    model.load_state_dict(checkpoint['state_dict'])
    return model