# Some code is inspired from https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
# and https://www.youtube.com/watch?v=BzcBsTou0C0&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=1 this playlist
# Had to use dev branch of skorch, conda skorch had problems: 
# pip install git+https://github.com/skorch-dev/skorch.git

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, Checkpoint, LoadInitState
from skorch.dataset import CVSplit, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using CPU .')
    device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self,
                 hidden_in=[1500, 450],
                 hidden_out=[450, 150],
                 num_classes=2,
                 dropout=0.5):

        super(Net, self).__init__()

        # Fully-connected input layer
        self.fc_input = nn.Linear(256, hidden_in[0])
        
        self.dropout_hidden = nn.Dropout(p=dropout)
        
        # fully-connected hidden layers 
        self.fc_hidden_list = nn.ModuleList([
            nn.Linear(hidden_in[i],
                      hidden_out[i])
            for i in range(len(hidden_in))
        ])
        
  
        # Fully-connected layer and Dropout
        self.fc_output = nn.Linear(hidden_out[-1], num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
       
        # Apply Fully-connected input layers and ReLU. Output shape: (b, output size)
        x_fc = F.relu(self.fc_input(input_ids.float()))
        
        # Apply dropout hidden. Output shape: (same as previous)
        #x_fc = self.dropout_hidden(x_fc)
        
        # Apply Fully-connected hidden layers and ReLU. Output shape: (b, hidden_out[i])
        for fc_hidden in self.fc_hidden_list:
            x_fc = F.relu(fc_hidden(x_fc))
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc_output(self.dropout(x_fc))
        
        return logits


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    # random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def predict(text, vocab, model, max_len=256):
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
    
    # Rescale input values
    input_id = input_id/69

    # Compute logits
    logits = model.forward(input_id)

    print(f"logits: {logits}")
    
    # Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    
    # Compute prediction
    y_pred = model.predict(input_id)
    
    print(f"softmax: {probs}")
    print(f"prediction: {y_pred}")

    print(f"This domain is {probs[1] * 100:.2f}% DGA.")
    print(f"Domain: {text}")
    # print(f"DNS query encoded: {text2idx}")
    print("-"*60)
    

def main():
    # Load training data
    training_data_X = torch.tensor(np.load(
        "../Dataset/dataset_NN/4.0/4.4/dataset_4.4/training_data_X.npy", allow_pickle=True))
    training_data_y = torch.tensor(np.load(
        "../Dataset/dataset_NN/4.0/4.4/dataset_4.4/training_data_y.npy", allow_pickle=True).astype(int))
    
    # Rescale X train
    training_data_X = training_data_X/69
    
    # Set split configuration
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # split training_data into train and validation sets
    # train is now 75% of the entire data set
    train_X, test_X, train_y, test_y = train_test_split(training_data_X, training_data_y, test_size=1 - train_ratio, shuffle=False)

    # Split train set into train and validation sets
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False)
    
    # Validation/train data split
    #VAL_PCT = 0.1
    #val_size = int(len(training_data_X)*VAL_PCT)

    # Split train data
    #train_X = training_data_X[:-val_size]
    #train_y = training_data_y[:-val_size]
    
    # Turn the labels from one hot to scalar: [0,1] or [1,0] to [0] or [1]
    # Loss function CrossEntropyLoss only takes scalar
    # Remove line below if loss function require one hot
    #train_y = torch.argmax(train_y, 1)

    # Split validation data
    #val_X = training_data_X[-val_size:]
    #val_y = training_data_y[-val_size:]
    
    # Turn the labels from one hot to scalar: [0,1] or [1,0] to [0] or [1]
    # Loss function CrossEntropyLoss only takes scalar
    # Remove line below if loss function require one hot
    #val_y = torch.argmax(val_y, 1)

    # Load dictonary/vocabulary which holds scalar represenation of characters in qname
    # used previously in preprocessing, example:
    # {'UNK': 0, 'a': 1, ... '}': 69}, uncomment print to see full
    vocab = np.load(
        "../Dataset/dataset_NN/4.0/4.4/dataset_4.4/dict.npy", allow_pickle=True).item()

    # Specify validation set
    val_dataset = Dataset(val_X, val_y)

    # Specify callbacks and checkpoints
    cp = Checkpoint(monitor='valid_acc_best', dirname='exp1')
    callbacks = [
        ('early_stop', EarlyStopping(monitor='valid_acc', patience=5, lower_is_better=False)),
        cp
    ]

    net = NeuralNetClassifier(
        # Module
        module=Net,
        module__hidden_in=[2000, 1000, 500],
        module__hidden_out=[1000, 500, 150],
        module__num_classes=2,
        module__dropout=0.2,
        # Optimizer
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adadelta,
        optimizer__lr=0.25,
        optimizer__rho=0.95,
        # Others
        max_epochs=10,
        batch_size=50,
        train_split=predefined_split(val_dataset),
        iterator_train__shuffle=True,
        warm_start=False,
        callbacks=callbacks,
        device=device
    )

    set_seed(42)
    _ = net.fit(np.array(train_X), train_y)

    valid_acc_best = np.max(net.history[:, 'valid_acc'])
    print(f"Training complete! Best accuracy: {valid_acc_best * 100:.2f}%")

    # Load parameters from checkpoint
    net.load_params(checkpoint=cp)

    print("\n")
    print("Start predict...")
    print("-"*60)

    predict("denon.jp",
            vocab=vocab, model=net, max_len=256)
    predict("andfeatencouragemission.com",
            vocab=vocab, model=net, max_len=256)
    predict("boshe-tk.ru",
            vocab=vocab, model=net, max_len=256)
    predict("ttxopkptonpczmo.biz",
            vocab=vocab, model=net, max_len=256)

    # print("\n")
    # print("Dictonary")
    # print("-"*60)
    # print(f"{vocab}")
    
    y_pred = net.predict(test_X)
    print(classification_report(test_y, y_pred))


if __name__ == "__main__":
    main()