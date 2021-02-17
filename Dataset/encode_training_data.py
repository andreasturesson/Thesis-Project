import numpy as np
import pandas as pd
import re
import csv
import os

# Inspired by https://github.com/netrack/learn/blob/master/dns/cnn.py


class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """
    def __init__(self, data_source, alphabet,
                 input_size=256, num_of_classes=2, data_size=1000):
        """
        Initialization of a Data object.
        Args:
            data_source (str): Raw data file path.
            alphabet (str): Alphabet of characters to index (possible characters in qname).
            input_size (int): Size of input features (max qname size).
            num_of_classes (int): Number of classes in data (1 or 0 in this case, true/false).
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        self.dict['UNK'] = 0
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.data_source = data_source
        self.data_size = data_size

    def load_data(self):
        """
        Load raw data from the source file into data variable.
        Returns: None.
        """
        data = []
        with open(self.data_source, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1]:
                    txt = txt + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                    #txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                if row[0] != "label":
                    data.append((int(row[0]), txt))  # format: (label, text)
        self.data = np.array(data)[:self.data_size]
        print("Data loaded from " + self.data_source)

    def get_dict(self):
        """returns dictionary, alphabet chars with paired int value.

        Returns:
            dict: dictionary.
        """
        return self.dict

    def get_none_encoded_data(self):
        """return data before encoding, i.e., qnames orginal form with 0 or 1 label.

        Returns:
            (np.array): unprocessed data.
        """
        return self.data

    def get_all_data(self):
        """Return all loaded data from data variable.

        Returns:
            (np.ndarray): Data transformed from raw to integer
            form, labeled is transformed from 0 or 1 to one-hot, i.e., [0,1] or [1,0].
        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            #c = int(c) - 1
            #classes.append(one_hot[c])
            classes.append(c)
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """Convert a string to character to integer based on character dictionary
        index placement.

        Args:
            s (str): String to be converted to integer from.
        Returns:
            str2idx: (np.ndarray): Indexes of characters in s.
        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
            # else, 'UNK' return str2idx all elements are zero
        return str2idx


def process_data(file):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    file_type = file.split("_")[0]

    dataframe = pd.read_csv(file) 
    if 'family' in dataframe.columns:
        family = dataframe[['family', 'label_multiclass']].to_numpy()
        np.save(f"dataset_NN/{file_type}_family.npy", family)

    input_file = open(file, "r+")
    reader_file = csv.reader(input_file)
    data_size = len(list(reader_file))-1

    data = Data(file, alphabet, 256, 2, data_size)
    data.load_data()
    data_X, data_y = data.get_all_data()

    dict = data.get_dict()
    none_encoded_data = data.get_none_encoded_data()

    # Save encoded and non encoded data aswell as the dictionary used to encode.
    # training_data_X = encoded qname
    # training_data_y = encoded label
    # training_none_encoded_data = qname and label non encoded
    # dict = dictionary for encoding
    np.save(f"dataset_NN/{file_type}_data_X.npy", data_X)
    np.save(f"dataset_NN/{file_type}_data_y.npy", data_y)
    np.save("dataset_NN/none_encoded_data", none_encoded_data)
    np.save("dataset_NN/dict.npy", dict)

    # Print example of encoding
    dict = np.load("dataset_NN/dict.npy", allow_pickle=True)
    none_encoded_data = np.load("dataset_NN/none_encoded_data.npy", allow_pickle=True)
    family = np.load(f"dataset_NN/{file_type}_family.npy", allow_pickle=True)
    print("Printing encoding exmaple:")
    print("-"*60)
    print(f"qname before encoding: {none_encoded_data[0][1]}")
    print("qname after encoding:")
    print(f"{data_X[0]} \n")
    print(f"label before encoding: {none_encoded_data[0][0]}")
    print(f"label after encoding: {data_y[0]} \n")
    print(f"family: {family[0]} \n")
    print("Dictionary used to encode:")
    print(dict)
    print("-"*60)
    print("\n")


def main():
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    GAN_train_file = "GANtrain_data.csv"
    GAN_test_file = "GANtest_data.csv"

    dataframe = pd.read_csv(train_file)
    dataframe = dataframe.loc[dataframe['label'] == 0]
    dataframe.to_csv(GAN_train_file, index=False, header=True)

    dataframe = pd.read_csv(test_file)
    dataframe = dataframe.loc[dataframe['label'] == 0]
    dataframe.to_csv(GAN_test_file, index=False, header=True)

    process_data(train_file)
    process_data(test_file)
    process_data(GAN_train_file)
    process_data(GAN_test_file)

    os.remove(GAN_train_file)
    os.remove(GAN_test_file)


if __name__ == "__main__":
    main()
