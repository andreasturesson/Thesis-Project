# Some code is inspired from https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
# and https://www.youtube.com/watch?v=BzcBsTou0C0&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=1 this playlist

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import helpfunc as hf

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_inputs, train_labels, val_inputs, val_labels = hf.load_trainset("Dataset/3.2/training_data_X.npy",
                                                                      "Dataset/3.2/training_data_y.npy")

train_dataloader, val_dataloader = hf.data_loader(train_inputs, train_labels, val_inputs, val_labels)

MODEL_NAME = f"model-[14-10-7-5]-[256-256-256-256]-{int(time.time())}"
print(MODEL_NAME)


class CNN(nn.Module):

    def __init__(self,
                 vocab_size=70,
                 embed_dim=128,
                 filter_sizes=[14, 10, 7, 5],
                 num_filters=[256, 256, 256, 256],
                 dropout=0.5):

        super(CNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters[i],
                                                    kernel_size=filter_sizes[i])
                                          for i in range(len(filter_sizes))])

        self.fc = nn.Linear(np.sum(num_filters), 2)     # Sum of outputs form convs, number of outcomes
        self.drop = nn.Dropout(dropout)     # Percentage of ignored nodes

    def forward(self, input):

        embed = self.embedding(input)
        embed = embed.permute(0, 2, 1)
        convs = [F.relu(conv1d(embed)) for conv1d in self.conv1d_list]
        pools = [F.max_pool1d(x, kernel_size=x.shape[2]) for x in convs]
        fc = torch.cat([x.squeeze(dim=2) for x in pools], dim=1)
        done = self.fc(self.drop(fc))
        return done


def train(model, optim, train_data, val_data=None, epochs=1):
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | "
          f"{'Val Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    best_accuracy = 0
    with open("model.log", "a") as f:
        for epoch in range(epochs):
            t0_epoch = time.time()
            total_loss = 0
            model.train()

            for i, data in enumerate(train_data):
                batch, labels = data
                model.zero_grad()
                outputs = model(batch.type(torch.LongTensor).to(device))
                preds = torch.argmax(outputs, dim=1).flatten()
                acc = (preds.cpu() == labels.cpu()).numpy().mean() * 100
                loss = loss_function(outputs, labels.type(torch.LongTensor).to(device))

                total_loss += loss.item()
                loss.backward()
                optim.step()

                if i % 100 == 0:
                    val_loss, val_accuracy = test(model, val_dataloader)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},"
                            f"{round(float(val_accuracy),2)},{round(float(val_loss),4)}\n")

            avg_train_loss = total_loss / len(train_dataloader)

            if val_data is not None:
                val_loss, val_accuracy = evaluate(model, val_dataloader)
                time_elapsed = time.time() - t0_epoch
                print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f} | "
                      f"{val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")


def evaluate(model, val_data):
    model.eval()

    val_accuracy = []
    val_loss = []

    for batch in val_data:
        batch, labels = tuple(t for t in batch)

        with torch.no_grad():
            outputs = model(batch.to(device))
        loss = loss_function(outputs, labels.type(torch.LongTensor).to(device))
        val_loss.append(loss.item())

        preds = torch.argmax(outputs, dim=1).flatten()

        accuracy = (preds.cpu() == labels.cpu()).numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def test(model, val_data):
    random_index = np.random.randint(len(val_data))
    for i, data in enumerate(val_data):
        if i == random_index:
            batch, labels = tuple(t for t in data)

    with torch.no_grad():
        outputs = model(batch.to(device))
        preds = torch.argmax(outputs, dim=1).flatten()
        acc = (preds.cpu() == labels.cpu()).numpy().mean() * 100
        loss = loss_function(outputs, labels.type(torch.LongTensor).to(device))
    return loss, acc


net = CNN()
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
train(net, optimizer, train_dataloader, val_dataloader, epochs=10)
