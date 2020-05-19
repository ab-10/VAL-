import torch.nn as nn

LATENT_CODE_SIZE = 128
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
            nn.Conv3d(in_channels = 1, out_channels = 1 * amcm, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(1 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 1 * amcm, out_channels = 2 * amcm, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(2 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 2 * amcm, out_channels = 4 * amcm, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(4 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 4 * amcm, out_channels = LATENT_CODE_SIZE * 2, kernel_size = 4, stride = 1),
            nn.BatchNorm3d(LATENT_CODE_SIZE * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.Linear(in_features = LATENT_CODE_SIZE * 2, out_features=LATENT_CODE_SIZE)
        ))
        self.cuda()

    def forward(self, x):
        return self.encoder(x)
