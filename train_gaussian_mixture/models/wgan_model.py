import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, nc):
        super(Generator, self).__init__()

        net = nn.Sequential(
            nn.Linear(nz, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, nc),
        )
        self.net = net
        self.nc = nc
        self.nz = nz
    
    def forward(self, input):
        output = self.net(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        
        net = nn.Sequential(
            nn.Linear(nc, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
        )
        self.net = net
        self.nc = nc

    def forward(self, input):
        output = self.net(input)
        return output