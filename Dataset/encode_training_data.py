import numpy as np
import re
import csv

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
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
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
            c = int(c) - 1
            classes.append(one_hot[c])
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


def main():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    input_file = open("training_data.csv", "r+")
    reader_file = csv.reader(input_file)
    data_size = len(list(reader_file))-1

    training_data = Data("training_data.csv", alphabet, 256, 2, data_size)
    training_data.load_data()
    training_data_X, training_data_y = training_data.get_all_data()

    dict = training_data.get_dict()
    training_none_encoded_data = training_data.get_none_encoded_data()

    # Save encoded and non encoded data aswell as the dictionary used to encode.
    # training_data_X = encoded qname
    # training_data_y = encoded label
    # training_none_encoded_data = qname and label non encoded
    # dict = dictionary for encoding
    np.save("dataset_NN/training_data_X.npy", training_data_X)
    np.save("dataset_NN/training_data_y.npy", training_data_y)
    np.save("dataset_NN/training_none_encoded_data", training_none_encoded_data)
    np.save("dataset_NN/dict.npy", dict)

    # Print example of encoding
    dict = np.load("dataset_NN/dict.npy", allow_pickle=True)
    none_encoded_data = np.load("dataset_NN/training_none_encoded_data.npy", allow_pickle=True)
    print("Printing encoding exmaple:")
    print("-"*60)
    print(f"qname before encoding: {none_encoded_data[1][1]}")
    print("qname after encoding:")
    print(f"{training_data_X[1]} \n")
    print(f"label before encoding: {none_encoded_data[1][0]}")
    print(f"label after encoding: {training_data_y[1]} \n")
    print("Dictionary used to encode:")
    print(dict)


if __name__ == "__main__":
    main()
