import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#GANtrain_data_X = np.load("dataset_NN/GANtrain_data_X.npy", allow_pickle=True)
#none_encoded_data = np.load("dataset_NN/GANtest_data_X.npy", allow_pickle=True)

train_X = torch.tensor(np.load(
        "../Dataset/dataset_NN/5.0/5.4/dataset/train_data_X.npy", allow_pickle=True))
#train_y = torch.tensor(np.load(
#        "../Dataset/dataset_NN/5.0/5.4/dataset/train_data_y.npy", allow_pickle=True).astype(int))
    
enc = OneHotEncoder(handle_unknown = 'ignore') #ignore tells the encoder to ignore new categories by encoding them with 0's
enc.fit(train_X)
onehot_train_X = enc.transform(train_X).toarray()
print(onehot_train_X.shape)
invers = enc.inverse_transform(onehot_train_X)
print(invers[2]) 

""" # Set split configuration
train_ratio = 0.75
validation_ratio = 0.05
test_ratio = 0.10 """

# split train_data into train and validation sets
# train is now 75% of the entire data set
#train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=1 - validation_ratio, shuffle=False)
#val_X, train_X, val_y, train_y = train_test_split(train_X, train_y, test_size=1 - validation_ratio, shuffle=False)

#print(train_X.shape)
#print(val_X.shape)

""" # One hot encode the integer encoded data
onehot_X = []
for domain in GANtrain_data_X:
    # one hot encode
    onehot_encoded_domain = list()
    for domain_char in domain:
        letter = [0 for _ in range(69)]
        letter[domain_char] = 1
        onehot_encoded_domain.append(letter)
    onehot_X.append(onehot_encoded_domain)  """
    
""" onehot_train_X = []
for domain in train_X:
    # one hot encode
    onehot_encoded_domain = list()
    for domain_char in domain:
        letter = [0 for _ in range(69)]
        letter[domain_char] = 1
        onehot_encoded_domain.append(letter)
    #print(onehot_encoded_domain)
    #print(len(onehot_encoded_domain))
    # invert encoding
    #string_inv = ""
    #for value in onehot_encoded_domain:
        #inverted = inv_dict[argmax(onehot_encoded[3])]
    #    inverted = inv_dict[argmax(value)]
    #    if inverted == 'UNK':
    #        continue
    #    string_inv += inverted
    #print(string_inv[::-1])
    onehot_train_X.append(onehot_encoded_domain) """

#print(FUCK)
# Save one hot encoded data
#np.save("GANtrain_onehot_data_X.npy", onehot_X) 