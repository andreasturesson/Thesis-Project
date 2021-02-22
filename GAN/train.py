import torch
import torch.optim as optim
from model import Discriminator, Generator, initialize_weights
from helpfunc import load_trainset, data_loader, set_seed
import numpy as np


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5  # 0.00005 0.001  could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
NOISE_DIM = 100
EMBED_DIM = 20
NUM_EPOCHS = 5
FEATURES_DISC = [100]
FILTERS_DISC = [3]
VOCAB_SIZE = 69
WEIGHT_CLIP = 0.01
CRITIC_ITER = 5

train_inputs, train_labels, val_inputs, val_labels = load_trainset("Dataset/GANtrain_data_X.npy",
                                                                   "Dataset/GANtrain_data_y.npy")
train_dataloader, val_dataloader = data_loader(train_inputs, train_labels, val_inputs, val_labels, BATCH_SIZE)
dict = np.load("Dataset/dict.npy", allow_pickle=True).item()
inv_dict = {v: k for k, v in dict.items()}
set_seed(seed_value=420)

gen = Generator(NOISE_DIM, VOCAB_SIZE, EMBED_DIM).to(device)
disc = Discriminator(VOCAB_SIZE, EMBED_DIM, FEATURES_DISC, FILTERS_DISC).to(device)
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)
# initialize_weights(gen)
# initialize_weights(disc)

gen.train()
disc.train()
with open("model.log", "a") as f:
    for epoch in range(NUM_EPOCHS):
        for idx, (real, _) in enumerate(train_dataloader):
            real_data = real.to(device)

            for _ in range(CRITIC_ITER):
                #noise = (torch.sigmoid(torch.randn(64, NOISE_DIM))*69).type(torch.LongTensor).to(device)
                noise = (torch.FloatTensor(64, NOISE_DIM).uniform_(0, 69)).type(torch.LongTensor).to(device)
                # fixa härifrån: https://pytorch.org/docs/stable/generated/torch.multinomial.html
                fake_data = gen(noise).to(device)
                disc_real = disc(real_data).reshape(-1)
                disc_fake = disc(real_data).reshape(-1)
                disc_loss = -(torch.mean(disc_real) - torch.mean(disc_fake))    # Minimize this
                disc.zero_grad()
                disc_loss.backward(retain_graph=True)
                opt_disc.step()

                # Enforce lipschitz constraint, info found in wgan paper
                for p in disc.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            output = disc(fake_data).reshape(-1)
            gen_loss = -torch.mean(output)
            gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            if idx % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {idx}/{len(train_dataloader)}] "
                    f"Loss D: {disc_loss:.8f}, Loss G: {gen_loss:.8f}"
                )

                word = ""
                for char in fake_data[0]:
                    if char.type(torch.IntTensor).item() is not 37:
                        character = inv_dict.get(char.type(torch.IntTensor).item())
                        if not character == "UNK":
                            word += character
                f.write(f"[Epoch: {epoch+1} Batch: {idx} Generated domain name: {word} \t [len: {len(word)}]\n")
                f.flush()
