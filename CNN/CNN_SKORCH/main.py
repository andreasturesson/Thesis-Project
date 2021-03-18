# Some code is inspired from https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
# and https://www.youtube.com/watch?v=BzcBsTou0C0&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=1 this playlist
# Had to use dev branch of skorch, conda skorch had problems: 
# pip install git+https://github.com/skorch-dev/skorch.git

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
#from keras.utils import to_categorical

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
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        
        # Get embeddings from `x`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped))
                       for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))
        
        return logits


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    # random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def predict(text, vocab, net, max_len=75):
    """Predict probability that a domain is DGA."""

    # Tokenize to single chars then encode chars to numerical values
    # based on dictionary key: {'UNK': 0, 'a': 1, ... '}': 69}
    text = text.lower()
    max_length = min(len(text), max_len)
    text2idx = np.zeros(max_len, dtype='int64')
    for i in range(1, max_length + 1):
        c = text[-i]
        if c in vocab:
            text2idx[i - 1] = vocab[c]
    
    # Convert to PyTorch tensors
    input_id = torch.tensor(text2idx).unsqueeze(dim=0)
    
    # Compute logits
    logits = net.forward(input_id)

    print(f"logits: {logits}")
    
    # Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    
    # Compute prediction
    y_pred = net.predict(input_id)
    
    print(f"softmax: {probs}")
    print(f"prediction: {y_pred}")

    print(f"This domain is {probs[1] * 100:.2f}% DGA.")
    print(f"Domain: {text}")
    # print(f"DNS query encoded: {text2idx}")
    print("-"*60)


def random_search(net, train_X, train_y):
    """Search for optimal hyperparameters."""
    
    
    params = {
        'batch_size': [10, 25, 50],
        'max_epochs': [5, 10, 25, 50],
        'module__dropout': [0.0, 0.2, 0.4, 0.6],
        'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    }
    
    #SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
    #gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')
    #gs = RandomizedSearchCV(estimator = net, param_distributions = params, n_iter = 2, cv = 2, verbose=0, random_state=100 , n_jobs = -1)
    
    # this will train 2 models over 3 folds of cross validation (2*3 models total)
    gs = RandomizedSearchCV(net, params, n_iter=60, random_state=2, cv=2)

    gs.fit(train_X, train_y)
    
    # Print result as table
    print("\n")
    print("Random search result:")
    df = pd.DataFrame(gs.cv_results_)
    df = df[['param_optimizer__lr','param_batch_size','param_max_epochs','param_module__dropout','mean_test_score','rank_test_score']]
    df.to_csv("grind_search.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("\n")
    
    # Print best score
    print("Best score:")
    print("-"*60)
    print(gs.best_score_, gs.best_params_)
    

def main():
    # Load train data
    train_X = torch.tensor(np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/train_data_X.npy", allow_pickle=True))
    train_y = torch.tensor(np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/train_data_y.npy", allow_pickle=True).astype(int))
    
    # Load test data
    test_X = torch.tensor(np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/test_data_X.npy", allow_pickle=True))
    test_y = torch.tensor(np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/test_data_y.npy", allow_pickle=True).astype(int))
    
    # Set split configuration
    validation_ratio = 0.10

    # split train_data into train and validation sets
    # train is now 90% of the entire data set
    val_X, train_X, val_y, train_y = train_test_split(train_X, train_y, test_size=1 - validation_ratio, shuffle=False)
    
    # Turn the labels from one hot to scalar: [0,1] or [1,0] to [0] or [1]
    # Loss function CrossEntropyLoss only takes scalar
    # Remove line below if loss function require one hot
    #train_y = torch.argmax(train_y, 1)
    #val_y = torch.argmax(val_y, 1)
    #test_y = torch.argmax(val_y, 1)


    # Load dictonary/vocabulary which holds scalar represenation of characters in qname
    # used previously in preprocessing, example:
    # {'UNK': 0, 'a': 1, ... '}': 69}, uncomment print to see full
    dict = np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/dict.npy", allow_pickle=True).item()
    
    # Load the families for the train set
    df = pd.read_csv("../Dataset/dataset_ensemble/6.0/6.5/test_family.csv")

    print(df.family[0])
    
    inv_dict = {v: k for k, v in dict.items()}
    
    word = ''
    for char in test_X[0]:
            character = inv_dict.get(char.type(torch.IntTensor).item())
            if not character == "UNK":
                word += character
    print(word[::-1])

    
    # Specify validation set
    val_dataset = Dataset(val_X, val_y)

    # Specify callbacks and checkpoints
    train_acc = EpochScoring(scoring='accuracy', on_train=True, 
                         name='train_acc', lower_is_better=False)

    cp = Checkpoint(monitor='valid_acc_best', dirname='exp1')
    callbacks = [
        ('early_stop', EarlyStopping(monitor='valid_acc', patience=20, lower_is_better=False)),
        cp,
        train_acc
    ]

    net = NeuralNetClassifier(
        # Module
        module=Net,
        module__vocab_size=69,
        module__embed_dim=128,
        #module__filter_sizes=[14, 7, 5, 3],
        #module__filter_sizes=[5, 4, 3, 2],
        module__filter_sizes=[14, 10, 7, 5],
        module__num_filters=[75, 75, 75, 75],
        module__num_classes=2,
        module__dropout=0.5,
        # Optimizer
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        optimizer__lr=0.001,
        #optimizer__rho=0.95,
        # Others
        max_epochs=100,
        batch_size=50,
        train_split=predefined_split(val_dataset),
        iterator_train__shuffle=True,
        warm_start=False,
        callbacks=callbacks,
        device=device
    )
      
    set_seed(42)
    
    # Preform grid seach i.e. find the best hyperparameters
    search = False
    if search:
        random_search(net=net, train_X=train_X, train_y=train_y)
        return
    
    _ = net.fit(np.array(train_X), train_y)

    valid_acc_best = np.max(net.history[:, 'valid_acc'])
    print(f"Training complete! Best accuracy: {valid_acc_best * 100:.2f}%")
    
    train_accuracy = net.history[:, 'train_acc']
    valid_accuracy = net.history[:, 'valid_acc']
    train_loss = net.history[:, 'train_loss']
    valid_loss = net.history[:, 'valid_loss']

    plt.figure(1)
    plt.plot(train_accuracy, 'o-', label='training')
    plt.plot(valid_accuracy, 'o-', label='validation')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.figure(2)
    plt.plot(train_loss, 'o-', label='training')
    plt.plot(valid_loss, 'o-', label='validation')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.show()

    # Load parameters from checkpoint
    net.load_params(checkpoint=cp)

    print("\n")
    print("Start predict...")
    print("-"*60)

    predict("denon.jp",
            vocab=dict, net=net, max_len=75)
    predict("andfeatencouragemission.com",
            vocab=dict, net=net, max_len=75)
    predict("boshe-tk.ru",
            vocab=dict, net=net, max_len=75)
    predict("ttxopkptonpczmo.biz",
            vocab=dict, net=net, max_len=75)

    # print("\n")
    # print("Dictonary")
    # print("-"*60)
    # print(f"{vocab}")
    
    y_pred = net.predict(test_X)
    print(classification_report(test_y, y_pred)) 
 

if __name__ == "__main__":
    main()