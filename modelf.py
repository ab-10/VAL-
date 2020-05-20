import torch
import torch.nn as nn

LATENT_CODE_SIZE = 128 # size of the Z vector
SDF_NET_BREADTH = 256 # size of the w vector

amcm = 24 # Autoencoder Model Complexity Multiplier

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class ModelF(nn.Module):
    def __init__(self):
        super(ModelF, self).__init__()
        self.add_module('encoder', nn.Sequential(
            nn.Linear(in_features = 3 + LATENT_CODE_SIZE, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True) 
        ))
        self.cuda()

    def forward(self, points, latent_codes):
        x = torch.cat((points, latent_codes), dim=1)
        x = x.cuda()
        return self.encoder(x)
