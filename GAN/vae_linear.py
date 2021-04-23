import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, features_dim, embed_dim):
        super(VAE, self).__init__()
        self.features = features_dim
        self.embed_dim = embed_dim

        # encoder
        self.enc1 = nn.Linear(in_features=69, out_features=100)
        self.enc2 = nn.Linear(in_features=100, out_features=50)
        self.enc3 = nn.Linear(in_features=50, out_features=self.features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=self.features, out_features=50)
        self.dec2 = nn.Linear(in_features=50, out_features=100)
        self.dec3 = nn.Linear(in_features=100, out_features=69)


    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mean + (eps * std) # sampling as if coming from the input space
        return sample #.repeat(1, 75).view(sample.shape[0], 75, self.features) # used for lstm

    def forward(self, x):
        # encoding
        [batch_size, word_len, encoding_dim] = x.size()
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # decoding
        x = F.relu(self.dec1(z.view(batch_size, word_len, -1)))
        x = F.relu(self.dec2(x))
        reconstruction = self.dec3(x) # softmax för one hot
        return reconstruction, mu, log_var
