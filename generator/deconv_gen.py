# Simple DCGAN
import torch
import torch.nn as nn
from functools import reduce
from operator import mul

import pdb

class Generator(nn.Module):
    def __init__(self, mapping, shapes, z_shape, dropout):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        layers = len(mapping)

        self.active = nn.ReLU(True)

        #add conv layers between
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_size, 256, (3, 4), 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*8) x 3 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # state size. (ngf*4) x 6 x 8
            nn.ConvTranspose2d(128, layers, 4, 2, 1, bias=False),
            # state size. (ngf*2) x 12 x 16
        )

        self.output = nn.Softmax2d()

    def forward(self, z):
        x = z.reshape(-1, self.z_size, 1, 1)
        x = self.main(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels, dropout):
        super(Discriminator, self).__init__()
        
        # Number of filters in the first layer
        ndf = 64
        
        self.main = nn.Sequential(
            # Input size is (input_channels) x 12 x 16
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # State size: (ndf) x 6 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # State size: (ndf*2) x 3 x 4
            nn.Conv2d(ndf * 2, 1, 3, 1, 0, bias=False),
            # State size: 1 x 1 x 1
            nn.Sigmoid()  # Outputs a single probability (real vs. fake)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)