# Highway class is taken from https://github.com/kefirski/pytorch_Highway
# Used under license: MIT License Copyright (c) 2017 Daniil Gavrilov

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_features, filter_sizes):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_features[i],
                                                    kernel_size=filter_sizes[i])
                                          for i in range(len(filter_sizes))])
        self.fc = nn.Linear(np.sum(num_features), 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, input):
        embed = self.embedding(input)
        embed = embed.permute(0, 2, 1)
        convs = [F.relu(conv1d(embed)) for conv1d in self.conv1d_list]
        pools = [F.max_pool1d(x, kernel_size=x.shape[2]) for x in convs]
        fc = torch.cat([x.squeeze(dim=2) for x in pools], dim=1)
        done = self.fc(self.drop(fc))
        return done

# very bad
class Generator(nn.Module):
    def __init__(self, noise_dim, vocab_size=69, embed_dim=20):
        super(Generator, self).__init__()

        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        # self.lstm = nn.LSTM(embed_dim, 128)
        # self.conv = nn.Conv1d(128, noise_dim, 4)
        # self.fc = nn.Linear(noise_dim, 75)

        # embedding_dim, kernel_size taken from paper but out_channels is improvised
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, 20)  # output here is ?
        self.highway = Highway(20, 10, torch.nn.functional.relu)
        self.convsize2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=2)
        self.convsize3 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3)
        self.linear = nn.Linear(1300, 75)


    def forward(self, input):
        embed = self.embedding(input)
        lstm, (hidden, cell) = self.lstm(embed)
        highway = self.highway(lstm)
        highway = highway.permute(0, 2, 1)
        conv2 = F.relu(self.convsize2(highway)) # output: 64, 20, 99
        conv3 = F.relu(self.convsize3(highway)) # output: 64, 10, 98
        #print(conv2.shape)
        #print(conv3.shape)
        conv2 = F.max_pool1d(conv2, kernel_size=2).view(64, 20*49)
        conv3 = F.max_pool1d(conv3, kernel_size=3).view(64, 10*32)
        #print(conv2.shape)
        #print(conv3.shape)
        convs = torch.cat((conv2, conv3), dim=1)
        done = torch.sigmoid(self.linear(convs))
        return (done*69).type(torch.LongTensor)

        # embed = self.embedding(input)
        # lstm, (hidden, cell) = self.lstm(embed)
        # lstm = lstm.permute(0, 2, 1)
        # conv = F.relu(self.conv(lstm))
        # conv = F.max_pool1d(conv, kernel_size=conv.shape[2])
        # done = torch.sigmoid(self.fc(conv.view(64, 256)))
        # return (done*69).type(torch.LongTensor)  # (self.net(input)*69).squeeze(dim=2).type(torch.LongTensor)

def test():
    gen = Generator(noise_dim=100, vocab_size=69, embed_dim=20)
    disc = Discriminator(vocab_size=69, embed_dim=128, num_features=[100,100,100], filter_sizes=[3,4,5])
    x = (torch.FloatTensor(64, 100).uniform_(0, 69)).type(torch.LongTensor)

    result = gen(x)
    # result2 = disc(result)

    dict = np.load(
        "Dataset/dict.npy", allow_pickle=True).item()

    print(dict)

    inv_dict = {v: k for k, v in dict.items()}
    for domain in x:
        word = ""
        for char in domain:
            character = inv_dict.get(char.type(torch.IntTensor).item())
            if not character == "UNK":
                word += character
        print(word)

    for query in result:
        word = ""
        for char in query:
            character = inv_dict.get(char.type(torch.IntTensor).item())
            if not character == "UNK":
                word += character
        print(word)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


#test()