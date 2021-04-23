import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from helpfunc import load_trainset, data_loader, set_seed


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense_disc = nn.Linear(in_features=4*18*17, out_features=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1, 75, 69, 1).permute(0, 3, 1, 2)))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        done = self.dense_disc(x.contiguous().view(-1, 4*18*17))
        return done


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.dense_gen = nn.Linear(in_features=noise_dim, out_features=4*18*17)
        self.transp1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, output_padding=(1, 0))
        self.transp2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, output_padding=1)

    def forward(self, x):
        x = F.relu(self.dense_gen(x).view(-1, 4, 18, 17))
        x = F.relu(self.transp1(x))
        done = self.transp2(x).squeeze()
        return done
