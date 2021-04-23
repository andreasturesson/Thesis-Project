import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, features_dim):
        super(VAE, self).__init__()
        self.features = features_dim

        # encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense_enc = nn.Linear(in_features=4*18*17, out_features=features_dim * 2)

        # decoder
        self.dense_dec = nn.Linear(in_features=features_dim, out_features=4*18*17)
        self.transp1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, output_padding=(1, 0))
        self.transp2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, output_padding=1)


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mean + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.conv1(x.view(-1, 75, 69, 1).permute(0, 3, 1, 2)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dense_enc(x.contiguous().view(-1, 4*18*17)).view(-1, 2, self.features)

        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dense_dec(z).view(-1, 4, 18, 17))
        x = F.relu(self.transp1(x))
        reconstruction = self.transp2(x).squeeze()
        return reconstruction, mu, log_var, z
