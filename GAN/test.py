import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import vae_simple
import vae_linear
import gan
import pandas as pd
from helpfunc import load_trainset, data_loader, set_seed, load_holdout_set, data_loader_holdout
from tqdm import tqdm
from sklearn.metrics import classification_report

LEARNING_RATE_VAE = 1e-4
LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DISC = 2e-5
EPOCHS = 100        # Pretrained for 400 epochs
BATCH_SIZE = 32
FEATURES_VAE = 128
EMBED_DIM = 100
TRAIN_VAE = False
TRAIN_GAN = False
MEAN = None
STD = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(13)  # 42 for training and 13 for holdout validation

train_inputs, train_labels, val_inputs, val_labels, \
disc_val_inputs, disc_val_labels = load_trainset("Dataset/GANtrain_data_X.npy", "Dataset/GANtrain_data_y.npy",
                                                 "Dataset/train_data_X.npy", "Dataset/train_data_y.npy")
train_dataloader, val_dataloader, disc_val_dataloader = data_loader(train_inputs, train_labels, val_inputs, val_labels,
                                                                    disc_val_inputs, disc_val_labels, BATCH_SIZE)

holdout_test_inputs, holdout_test_labels = load_holdout_set("Dataset/test_data_X.npy", "Dataset/test_data_y.npy")
holdout_test_dataloader = data_loader_holdout(holdout_test_inputs, holdout_test_labels, BATCH_SIZE)

dict = np.load("Dataset/dict.npy", allow_pickle=True).item()
inv_dict = {v: k for k, v in dict.items()}
print(inv_dict)

vae = vae_simple.VAE(features_dim=FEATURES_VAE).to(device)
optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE_VAE)
loss_func_vae = nn.CrossEntropyLoss(reduction="sum")  # mseloss       [64, 75] -> [64, 75, 69] cross entropy loss

gen = gan.Generator(FEATURES_VAE).to(device)
disc = gan.Discriminator().to(device)
optimizer_gen = optim.SGD(gen.parameters(), lr=LEARNING_RATE_GEN)
optimizer_disc = optim.SGD(disc.parameters(), lr=LEARNING_RATE_DISC)
loss_func_gan = nn.CrossEntropyLoss()


def final_loss(bce_loss, mu, logvar):
    return bce_loss + (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu))


def init_vae(vae):
    for m in vae.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def train_VAE(model, dataloader):
    init_vae(model)
    model.train()
    running_loss = 0.0
    for i, (data, _) in tqdm(enumerate(dataloader), total=int(len(train_inputs) / dataloader.batch_size)):
        data = nn.functional.one_hot(data, num_classes=69).type(torch.FloatTensor)
        optimizer_vae.zero_grad()
        reconstruction, mu, logvar, _ = model(data.to(device))  # .to(device)
        _, targets = data.max(dim=2)
        bce_loss = loss_func_vae(reconstruction.permute(0, 2, 1), targets.to(device))  # .to(device)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer_vae.step()
    train_loss = running_loss / len(dataloader.dataset)
    return train_loss, data[0], reconstruction[0]


def validate_VAE(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (data, _) in tqdm(enumerate(dataloader), total=int(len(val_inputs) / dataloader.batch_size)):
            data = nn.functional.one_hot(data, num_classes=69).type(torch.FloatTensor)
            reconstruction, mu, logvar, z = model(data.to(device))
            _, targets = data.max(dim=2)
            bce_loss = loss_func_vae(reconstruction.permute(0, 2, 1), targets.to(device))
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, data[0], reconstruction[0].type(torch.LongTensor), mu[0], logvar[0], z[0]


def init_gan(gen, vae):
    gen.dense_gen.weight.data = vae.dense_dec.weight.data
    gen.dense_gen.bias.data = vae.dense_dec.bias.data
    gen.transp1.weight.data = vae.transp1.weight.data
    gen.transp1.bias.data = vae.transp1.bias.data
    gen.transp2.weight.data = vae.transp2.weight.data
    gen.transp2.bias.data = vae.transp2.bias.data


def dom_to_str(domain):
    domain_str = ""
    for char in reversed(domain):
        character = inv_dict.get(char.item())
        if not character == "UNK":
            domain_str += character
    return domain_str


def train_GAN(dataloader):
    losses_gen = []
    losses_disc = []

    for batch_idx, (real, _) in tqdm(enumerate(dataloader), total=int(len(train_inputs) / dataloader.batch_size)):
        real = nn.functional.one_hot(real, num_classes=69).type(torch.FloatTensor)
        noise = torch.randn(BATCH_SIZE, FEATURES_VAE)   # TODO: maybe sample noise from autoencoder distribution
        # noise = torch.empty(BATCH_SIZE, FEATURES_VAE).normal_(mean=torch.mean(MEAN), std=torch.mean(STD))

        fake = gen(noise.to(device))  # [32, 75, 69]
        fake = torch.argmax(fake.permute(0, 2, 1), dim=1)
        fake = nn.functional.one_hot(fake, num_classes=69).type(torch.FloatTensor)

        # Train Discriminator:
        disc_real = disc(real.type(torch.FloatTensor).to(device))
        loss_disc_real = loss_func_gan(disc_real, torch.zeros(disc_real.shape[0]).type(torch.LongTensor).to(device))
        disc_fake = disc(fake.type(torch.FloatTensor).to(device).detach())
        loss_disc_fake = loss_func_gan(disc_fake, torch.ones(disc_fake.shape[0]).type(torch.LongTensor).to(device))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        losses_disc.append(loss_disc.cpu().detach().numpy())

        disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # Train Generator:
        output = disc(fake.type(torch.FloatTensor).to(device))
        loss_gen = loss_func_gan(output, torch.zeros(output.shape[0]).type(torch.LongTensor).to(device))
        losses_gen.append(loss_gen.cpu().detach().numpy())
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx % 100 == 0:
            print(f"\nGen loss: {loss_gen}")
            print(f"Disc loss real: {loss_disc_real}")
            print(f"Disc loss fake: {loss_disc_fake}")
            domain = dom_to_str(torch.argmax(fake[0], dim=1))
            print(f"Example domain: {domain}")

        if batch_idx % 1000 == 0:
            print(f"\nReal predictions: {torch.argmax(disc_real, dim=1)}")
            print(f"Fake predictions: {torch.argmax(disc_fake, dim=1)}")
    return np.mean(losses_gen), np.mean(losses_disc)


def eval_discriminator(dataloader):
    disc.eval()
    val_accuracy = []
    all_preds = []

    for batch_idx, (batch, labels) in enumerate(dataloader):
        batch = nn.functional.one_hot(batch, num_classes=69).type(torch.FloatTensor)
        with torch.no_grad():
            outputs = disc(batch.to(device))
        outputs = outputs.cpu()
        preds = torch.argmax(outputs, dim=1).flatten()
        [all_preds.append(x) for x in preds.numpy()]
        accuracy = (preds.cpu() == labels.cpu()).numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Print results to excel sheet for mcnemar test
    # Change labels to be correct with the validated dataset
    mc_nemars_test = pd.DataFrame({'actual': holdout_test_labels, 'predicted': all_preds}, columns=['actual', 'predicted'])
    mc_nemars_test.to_excel('Logs/mcNemars_Test1.xlsx', index=False, header=True)
    print(classification_report(holdout_test_labels, all_preds, digits=4))

    val_accuracy = np.mean(val_accuracy)
    return val_accuracy


if TRAIN_VAE:
    with open("Logs/train_vae.log", "a") as f:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1} of {EPOCHS}")
            train_epoch_loss, train_indom, train_outdom = train_VAE(vae, train_dataloader)

            train_outdom_str = dom_to_str(torch.argmax(train_outdom, dim=1))
            train_indom_str = dom_to_str(torch.argmax(train_indom, dim=1))

            val_epoch_loss, val_indom, val_outdom, MEAN, LOGVAR, z = validate_VAE(vae, val_dataloader)

            val_outdom_str = dom_to_str(torch.argmax(val_outdom, dim=1))
            val_indom_str = dom_to_str(torch.argmax(val_indom, dim=1))

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")
            print(f"In domain: {train_indom_str}")
            print(f"Out domain: {train_outdom_str}")
            print(f"In domain validation: {val_indom_str}")
            print(f"Out domain validation: {val_outdom_str}")
            f.write(f"Epoch: {epoch}, "
                    f"Train Loss: {train_epoch_loss:.4f}, "
                    f"Val Loss: {val_epoch_loss:.4f}, "
                    f"In domain: {train_indom_str}, "
                    f"Out domain: {train_outdom_str}, "
                    f"In domain validation: {val_indom_str}, "
                    f"Out domain validation: {val_outdom_str}\n")
            f.flush()

        torch.save(vae, "Model/vae.pt")

if not TRAIN_VAE:
    vae = torch.load("Model/vae.pt")
    _, val_indom, val_outdom, MEAN, logvar, z = validate_VAE(vae, val_dataloader)
    STD = torch.exp(0.5 * logvar)
    print(MEAN)
    print(STD)
    print(torch.empty(BATCH_SIZE, FEATURES_VAE).normal_(mean=torch.mean(MEAN), std=torch.mean(STD)))
    val_indom_str = dom_to_str(torch.argmax(val_indom, dim=1))
    val_outdom_str = dom_to_str(torch.argmax(val_outdom, dim=1))
    print(f"In domain validation: {val_indom_str}")
    print(f"Out domain validation: {val_outdom_str}")

if TRAIN_GAN:
    init_gan(gen, vae)
    gen.train()
    disc.train()
    print("\n --- Start training gan ---")
    with open("Logs/train_gan_2.log", "a") as f:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1} of {EPOCHS}")
            gen_loss, disc_loss = train_GAN(train_dataloader)
            torch.save(disc, "Model/disc.pt")
            eval_acc = eval_discriminator(disc_val_dataloader)
            print(f"\nDiscriminator validation accuracy: {eval_acc}")
            f.write(f"{epoch+1},{gen_loss},{disc_loss},{eval_acc}\n")
            f.flush()

if not TRAIN_GAN:
    disc = torch.load("Model/disc.pt")
    print(eval_discriminator(holdout_test_dataloader))

