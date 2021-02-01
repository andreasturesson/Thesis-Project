import numpy as np
import pandas as pd
import random
import matplotlib as mp

TEST_SIZE = 0.2
DATASET = "../../../Dataset/dataset_ensemble/2.0/dataset_2.1.csv"

def loadData(filename):
    dataframe = pd.read_csv(filename, header=None, low_memory=False)
    dataframe = dataframe.sample(frac=1)
    return dataframe

def trainTestSplit(dataframe, test_size):
    if isinstance(test_size, float):
        test_size = int(round(test_size * len(dataframe)))

    data_list = dataframe.values.tolist()
    testing_data = data_list[0:int(TEST_SIZE*len(data_list))]
    training_data = data_list[int(TEST_SIZE*len(data_list)+1):len(data_list)]

    return training_data, testing_data


def uniqueValues(rows, col):
    return set([row[col] for row in rows])

def isNum(value):
    return isinstance(value, int) or isinstance(value, float)

def classCount(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class dontApprove:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        value = example[self.column]
        if isNum(value):
            return value >= self.value
        else:
            return value == self.value


def partition(rows, DontApprove):
    true_rows, false_rows = [], []
    for row in rows:
        if DontApprove.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def giniImpurity(rows):
    counts = classCount(rows)
    impurity = 1
    for label in counts:
        probability_of_label = counts[label] / float(len(rows))
        impurity -= probability_of_label ** 2
    return impurity


def informationGain(left_node, right_node, current_impurity):
    probability = float(len(left_node)) / (len(right_node))
    return current_impurity - probability * giniImpurity(left_node) - (1 - probability) * giniImpurity(right_node)

# If the impurity is lower than the weight pick create a new question
def split(rows):
    highest_valued_gain = 0
    highest_valued_question = None
    current_impurity = giniImpurity(rows)
    number_of_features = len(rows[0])-1

    for columns in range(number_of_features):
        values = set([row[columns] for row in rows])
        for value in values:
            question = dontApprove(columns, value)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            else:
                gain = informationGain(true_rows, false_rows, current_impurity)
            if gain >= highest_valued_gain:
                highest_valued_gain, highest_valued_question = gain, question
    return highest_valued_gain, highest_valued_question

class Leaf:
    def __init__(self, rows):
        self.predictions = classCount(rows)



class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# Can fuck with values
def buildTree(rows):
    gain, question = split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = buildTree(true_rows)
    false_branch = buildTree(false_rows)
    return DecisionNode(question, true_branch, false_branch)

#fit function
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


if __name__ == '__main__':
    dataframe = loadData(DATASET)
    training_data, testing_data = trainTestSplit(dataframe, TEST_SIZE)
    tree = buildTree(training_data)