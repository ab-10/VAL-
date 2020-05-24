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

class ModelG(nn.Module):
    def __init__(self):
        super(ModelG, self).__init__()
        self.add_module('generator', nn.Sequential(
            # accepts x,y,z, and w
            nn.Linear(in_features =3+ SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            nn.Linear(in_features = SDF_NET_BREADTH, out_features = SDF_NET_BREADTH),
            nn.ReLU(inplace=True),

            # outputs SDF(x,y,z)
            nn.Linear(in_features = SDF_NET_BREADTH, out_features = 1),
            nn.ReLU(inplace=True) 
        ))
        self.cuda()

    def forward(self, p, w):
        # TODO: is it necessary to .cuda() at cat result?
        x = torch.cat((p.cuda(), w.cuda()), dim=1).cuda()
        return self.generator(x)
