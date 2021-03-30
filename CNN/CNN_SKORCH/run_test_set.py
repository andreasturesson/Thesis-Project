from pickle import TRUE
from skorch.callbacks.scoring import EpochScoring
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import argmax
from tabulate import tabulate
from matplotlib import pyplot as plt
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, Checkpoint, LoadInitState
from skorch.dataset import CVSplit, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using CPU .')
    device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self,
                 vocab_size=69,
                 embed_dim=128,
                 filter_sizes=[10, 7, 5, 3],
                 num_filters=[75, 75, 75, 75],
                 num_classes=2,
                 dropout=0.5):

        super(Net, self).__init__()
        # Embedding layer
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=0,
                                      max_norm=5.0)

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        #self.fc = nn.Linear(np.sum(num_filters), num_classes)
        
        self.fc_hidden = nn.Linear(np.sum(num_filters), 300)   #ÄNDRAD
        
        self.fc = nn.Linear(300, num_classes)  # ÄNDRAD
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # Get embeddings from `x`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()
        

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        #x_conv_list = [F.relu(conv1d(x_reshaped))               
        #               for conv1d in self.conv1d_list]
                
        # Apply CNN. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [conv1d(x_reshaped)              # ÄNDRAD
                       for conv1d in self.conv1d_list] # ÄNDRAD


        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]
        
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Hidden fully connected layer
        x_fc = F.relu(self.fc_hidden(self.dropout(x_fc)))  

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits

def set_seed(seed_value=13):
    """Set seed for reproducibility."""

    # random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
net = NeuralNetClassifier(
        # Module
        module=Net,
        module__vocab_size=69,
        module__embed_dim=128,
        module__filter_sizes=[6, 5, 4, 3, 2],
        module__num_filters=[75, 75, 75, 75, 75],
        module__num_classes=2,
        module__dropout=0.3,
        # Optimizer
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        optimizer__lr=0.001,
        #optimizer__rho=0.95,
        # Others
        max_epochs=100,
        batch_size=50,
        #train_split=predefined_split(val_dataset),
        iterator_train__shuffle=False,
        warm_start=False,
        #callbacks=callbacks,
        device=device
    )

# Load test data
test_X = torch.tensor(np.load(
    "../Dataset/dataset_NN/5.0/5.4/dataset/test_data_X.npy", allow_pickle=True))
test_y = torch.tensor(np.load(
    "../Dataset/dataset_NN/5.0/5.4/dataset/test_data_y.npy", allow_pickle=True).astype(int))
    
set_seed(13)

# Set split configuration
#validation_ratio = 0.25


# split train_data into train and validation sets
# train is now 90% of the entire data set
#val_X, train_X, val_y, train_y = train_test_split(train_X, train_y, test_size=1 - validation_ratio, shuffle=False, random_state=42)


cp = Checkpoint(monitor='valid_loss_best', dirname='exp1')

# Load parameters from checkpoint
net.load_params(checkpoint=cp)

train_accuracy = net.history[:, 'train_acc']
valid_accuracy = net.history[:, 'valid_acc']
train_loss = net.history[:, 'train_loss']
valid_loss = net.history[:, 'valid_loss']


plt.figure(1)
plt.plot(train_accuracy, label='training')
plt.plot(valid_accuracy,  label='validation')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.legend()
plt.figure(2)
plt.plot(train_loss, label='training')
plt.plot(valid_loss, label='validation')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.show()

y_pred = net.predict(test_X)

print(classification_report(test_y, y_pred, digits=4))

data = {'actual': test_y, 'predicted': y_pred}
 
df = pd.DataFrame(data)

print(df)
df['result'] = np.where(df['actual'] == df['predicted'], '1', '0')
df.to_csv('CNN_result_test.csv', index=False)
df.to_excel('CNN_result_test.xlsx', index=False)
