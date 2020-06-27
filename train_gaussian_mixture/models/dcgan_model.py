import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.net = net
        self.nc = nc

    def forward(self, input):
        output = self.net(input)
        return output.view(output.shape[0], -1)