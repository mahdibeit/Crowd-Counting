import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.bn = nn.BatchNorm2d(512)
        self.mlp_layer = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(512, 32, 1),
            nn.Conv2d(32, 32, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 1),
            nn.BatchNorm2d(512)
        )
        self.flatten = nn.Flatten()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.test = nn.Linear(512, 32)

    def forward(self, x):

        x = self.features(x) # torch.Size([1, 512, 64, 64])

        x_att = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x_att = self.flatten(x_att)
        x_att = self.mlp_layer(x_att).unsqueeze(2).unsqueeze(3)
        x_att = self.bn(x_att)
        x = (1 + F.sigmoid(x_att)) * x

        x = F.interpolate(x, scale_factor=2, mode='bilinear') # torch.Size([1, 512, 64, 64])
        x = self.reg_layer(x) # torch.Size([1, 512, 32, 32])
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

