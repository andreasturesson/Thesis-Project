import numpy as np
import pandas as pd
import random
import matplotlib as mp
from tqdm import tqdm

TEST_SIZE = 0.2
RANDOM_SUBSPACE = 15
FOREST_SIZE = 5
BOOTSTRAP_SAMPLE_SIZE = 0.333
DATASET = "../../../../Dataset/dataset_ensemble/2.0/dataset_2.1.csv"
EPOCH = 10
ACCURACY = []
ACCURACY_AVERAGE = 0
ACCURACY_BEST = 0
ACCURACY_WORST = 0

def writeResults():
    results = open("results.txt", "a")
    results.write("TEST_SIZE: %s \nRANDOM_SUBSPACE: %s \nFOREST_SIZE: %s \nBOOTSTRAP_SAMPLE_SIZE: %s \n"%(TEST_SIZE, RANDOM_SUBSPACE, FOREST_SIZE, BOOTSTRAP_SAMPLE_SIZE))
    results.write("DATASET: " + DATASET + "\n")
    results.write("EPOCS: " + str(EPOCH) + "\n\n")
    results.write("Accuracy: %s" %(ACCURACY))
    ACCURACY_WORST = ACCURACY[0]
    ACCURACY_BEST = ACCURACY[0]
    sum = 0
    for accu in ACCURACY:
        sum += accu
        if(ACCURACY_BEST < accu):
            ACCURACY_BEST = accu
        if(ACCURACY_WORST > accu):
            ACCURACY_WORST = accu
    ACCURACY_AVERAGE = sum / EPOCH

    results.write("\nAccuracy worst: %s\nAccuracy best: %s\nAccuracy average: %s" %(ACCURACY_WORST, ACCURACY_BEST, ACCURACY_AVERAGE))
    results.write("\n------------------------------------------------------------------------\n")
    results.close()

def loadData(filename):
    dataframe = pd.read_csv(filename, low_memory=False)
    dataframe = dataframe.sample(frac=1)
    return dataframe

def trainTestSplit(dataframe, test_size):
    data_list = dataframe.values.tolist()

    testing_data = data_list[0:round((int(TEST_SIZE*len(data_list))))]
    training_data = data_list[round(int(test_size*len(data_list))):len(data_list)]

    return training_data, testing_data

# array [[33% traning_data], [33% traning_data], [33% traning_data], [33% traning_data]]
def bootstrapTrainingData(size, training_data):
    training_bootstrap_sampels = []
    for i in range(size):
        training_bootstrap_sampels.append(random.choices(training_data, k=int(round(len(training_data)*BOOTSTRAP_SAMPLE_SIZE))))

    return training_bootstrap_sampels

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

class Question:
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
def split(rows, number_of_features_list):

    highest_valued_gain = 0
    highest_valued_question = None
    current_impurity = giniImpurity(rows)


    for columns in number_of_features_list:

        values = set([row[columns] for row in rows])
        for value in values:
            question = Question(columns, value)
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

def randomSubspaceSample(rows):
    number_of_features_list = []
    number_of_features = len(rows[0])-1
    for i in range(number_of_features):
        number_of_features_list.append(i)
    number_of_features_list = random.sample(number_of_features_list, k=RANDOM_SUBSPACE)
    return number_of_features_list

# Can fuck with values # don't do it
def buildTree(rows, number_of_features_list):

    gain, question = split(rows, number_of_features_list)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = buildTree(true_rows,number_of_features_list)
    false_branch = buildTree(false_rows,number_of_features_list)
    return DecisionNode(question, true_branch, false_branch)

#fit function
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def accuracy(details, forest, testing_data):
    bb, cc = 0, 0
    bb2, cc2 = 0,0

    for row in testing_data:
        bv,gv = 0,0
        bb += 1

        actual_value = row[-1]
        for tree in forest:
            prediciton = list(classify(row[0:-1], tree).keys())[0]

            if int(prediciton) == 1:
                bv += 1
            elif int(prediciton) == 0:
                gv += 1
            if (details):
                #print(row[0:-1])
                print(classify(row[0:-1], tree))
                print("Actual: %s. Predicted: %s" %(actual_value, prediciton))

        if gv >= bv:
            prediciton = 0
        else:
            prediciton = 1
        if int(prediciton) == int(actual_value):
            cc += 1

    accuracy = cc / bb * 100
    ACCURACY.append(accuracy)
    print("Accuracy is: %s Correct: %s Total: %s" %(accuracy, cc, bb), "\n")

def buildForest(bootstrap_training_data):
    forest = []
    for i in tqdm(range(FOREST_SIZE)):
        forest.append(buildTree(bootstrap_training_data[i], randomSubspaceSample(bootstrap_training_data[i])))
    return forest

if __name__ == '__main__':
    dataframe = loadData(DATASET)
    training_data, testing_data = trainTestSplit(dataframe, TEST_SIZE)

    for epoc in range(EPOCH):
        bootstrap_training_data = bootstrapTrainingData(FOREST_SIZE, training_data)
        forest = buildForest(bootstrap_training_data)
        accuracy(False, forest, testing_data)

    writeResults()




