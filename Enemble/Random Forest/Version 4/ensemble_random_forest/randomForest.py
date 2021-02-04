import pandas as pd
import random
from tqdm import tqdm

'''
Code is inspired or acquired from: Sebastian Mantey(youtube) and AI Sciences, AI Sciences Team(udemy)
Source: https://www.youtube.com/watch?v=WvmPnGmCaIM&list=PLPOTBrypY74y0DviMOagKRUhDdk0JyM_r (Obtained: 2021/02/01)
Source: https://www.udemy.com/course/decision-tree-and-random-forest-python-from-zero-to-hero/ (Obtained: 2021/01/20)
'''

class RandomForest:

    def __init__(self, DATASET, TEST_SIZE = 0.2, RANDOM_SUBSPACE = 10, FOREST_SIZE = 15, MAX_DEPTH = 1000, BOOTSTRAP_SAMPLE_SIZE = 0.33, EPOCH = 1, details =False):
        self.TEST_SIZE = TEST_SIZE
        self.RANDOM_SUBSPACE = RANDOM_SUBSPACE  # n/3  or Root(n)
        self.FOREST_SIZE = FOREST_SIZE
        self.MAX_DEPTH = MAX_DEPTH
        self.BOOTSTRAP_SAMPLE_SIZE = BOOTSTRAP_SAMPLE_SIZE
        self.EPOCH = EPOCH
        self.ACCURACY = []
        self.ACCURACY_AVERAGE = 0
        self.ACCURACY_BEST = 0
        self.ACCURACY_WORST = 0
        self.DATASET = DATASET
        self.details = details

        self.data_frame = self.loadData(self.DATASET)
        self.training_data, self.testing_data = self.trainTestSplit(self.data_frame, self.TEST_SIZE)

        for epoc in range(self.EPOCH):
            self.bootstrap_training_data = self.bootstrapTrainingData(self.FOREST_SIZE, self.training_data)
            self.forest = self.buildForest(self.bootstrap_training_data, self.MAX_DEPTH)
            self.accuracy(self.details, self.forest, self.testing_data)

        self.writeResults()

    def randomSubspaceSample(self, rows):
        number_of_features_list = []
        number_of_features = len(rows[0]) - 1
        for i in range(number_of_features):
            number_of_features_list.append(i)
        number_of_features_list = random.sample(number_of_features_list, k= self.RANDOM_SUBSPACE)
        return number_of_features_list

    def partition(self, rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):  # Returns True or False, fills up true/false lists
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def giniImpurity(self, rows):
        counts = self.classCount(rows)
        impurity = 1
        for label in counts:
            probability_of_label = counts[label] / float(len(rows))
            impurity -= probability_of_label ** 2
        return impurity

    def classCount(self, rows):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def accuracy(self, details, forest, testing_data):
        bb, correct_cal1, correct_cal2 = 0, 0, 0
        for row in testing_data:
            bv_average, gv_average = 0, 0
            bv_average_total, gv_average_total = 0, 0

            bb += 1
            actual_value = row[-1]
            for tree in forest:
                prediciton = list(self.classify(row[0:-1], tree).keys())[0]

                if int(prediciton) == 1:
                    bv_average += 1
                elif int(prediciton) == 0:
                    gv_average += 1

                if (self.classify(row[0:-1], tree).get(0) != None):
                    gv_average_total += self.classify(row[0:-1], tree).get(0)
                if (self.classify(row[0:-1], tree).get(1) != None):
                    bv_average_total += self.classify(row[0:-1], tree).get(1)
                if (details):
                    print(row[0:-1])
                    print(self.classify(row[0:-1], tree))
                    print("Actual: %s. Predicted: %s" % (actual_value, prediciton))

            if gv_average >= bv_average:
                prediciton_average = 0
            else:
                prediciton_average = 1
            if int(prediciton_average) == int(actual_value):
                correct_cal1 += 1

            if gv_average_total >= bv_average_total:
                prediciton_average_total = 0
            else:
                prediciton_average_total = 1
            if int(prediciton_average_total) == int(actual_value):
                correct_cal2 += 1

        accuracy_average = correct_cal1 / bb * 100
        accuracy_average_total = correct_cal2 / bb * 100
        self.ACCURACY.append(accuracy_average)
        print("Accuracy average decision: %s Correct: %s Total: %s" % (accuracy_average, correct_cal1, bb), "\n")
        print("Accuracy total average: %s Correct: %s Total: %s" % (accuracy_average_total, correct_cal2, bb), "\n")

    def informationGain(self, left_node, right_node, current_impurity):
        probability = float(len(left_node)) / (len(right_node))
        return current_impurity - probability * self.giniImpurity(left_node) - (1 - probability) * self.giniImpurity(right_node)

    def buildTree(self, rows, MAX_DEPTH):
        gain, question = self.split(rows)
        if 0 <= gain <= 0.0001 or MAX_DEPTH == 0:
            return Leaf(rows)
        true_rows, false_rows = self.partition(rows, question)
        true_branch = self.buildTree(true_rows, MAX_DEPTH - 1)
        false_branch = self.buildTree(false_rows, MAX_DEPTH - 1)
        return DecisionNode(question, true_branch, false_branch)

    def buildForest(self, bootstrap_training_data, MAX_DEPTH):
        forest = []
        for i in tqdm(range(self.FOREST_SIZE)):
            forest.append(self.buildTree(bootstrap_training_data[i], MAX_DEPTH))
        return forest

    def writeResults(self):
        results = open("results.txt", "a")
        results.write(
            "TEST_SIZE: %s \nRANDOM_SUBSPACE: %s \nFOREST_SIZE: %s \nMAX_DEPTH: %s \nBOOTSTRAP_SAMPLE_SIZE: %s \n" % (
                self.TEST_SIZE, self.RANDOM_SUBSPACE, self.FOREST_SIZE, self.MAX_DEPTH, self.BOOTSTRAP_SAMPLE_SIZE))
        results.write("DATASET: " + self.DATASET + "\n")
        results.write("EPOCS: " + str(self.EPOCH) + "\n\n")
        results.write("Accuracy: %s" % (self.ACCURACY))
        self.ACCURACY_WORST = self.ACCURACY[0]
        self.ACCURACY_BEST = self.ACCURACY[0]
        self.sum = 0
        for accu in self.ACCURACY:
            self.sum += accu
            if (self.ACCURACY_BEST < accu):
                self.ACCURACY_BEST = accu
            if (self.ACCURACY_WORST > accu):
                self.ACCURACY_WORST = accu
        self.ACCURACY_AVERAGE = self.sum / self.EPOCH

        results.write("\nAccuracy worst: %s\nAccuracy best: %s\nAccuracy average: %s" % (
            self.ACCURACY_WORST, self.ACCURACY_BEST, self.ACCURACY_AVERAGE))
        results.write("\n------------------------------------------------------------------------\n")
        results.close()

    def loadData(self, filename):
        self.DATASET = filename
        dataframe = pd.read_csv(filename, low_memory=False)
        dataframe = dataframe.sample(frac=1)
        return dataframe

    def trainTestSplit(self, dataframe, test_size):
        data_list = dataframe.values.tolist()

        testing_data = data_list[0:round((int(self.TEST_SIZE * len(data_list))))]
        training_data = data_list[round(int(test_size * len(data_list))):len(data_list)]

        return training_data, testing_data

    def bootstrapTrainingData(self, size, training_data):
        training_bootstrap_sampels = []
        for i in range(size):
            training_bootstrap_sampels.append(
                random.choices(training_data, k=int(round(len(training_data) * self.BOOTSTRAP_SAMPLE_SIZE))))

        return training_bootstrap_sampels

    def uniqueValues(self, rows, col):
        return set([row[col] for row in rows])

    def isNum(self, value):
        return isinstance(value, int) or isinstance(value, float)

    def split(self, rows):
        highest_valued_gain = 0
        highest_valued_question = None
        current_impurity = self.giniImpurity(rows)
        number_of_features_list = self.randomSubspaceSample(rows)

        for columns in number_of_features_list:

            values = set([row[columns] for row in rows])
            for value in values:
                question = Question(columns, value)
                true_rows, false_rows = self.partition(rows, question)
                if len(true_rows) == 0 or len(false_rows) == 0:

                    continue
                else:
                    gain = self.informationGain(true_rows, false_rows, current_impurity)

                if gain >= highest_valued_gain:
                    highest_valued_gain, highest_valued_question = gain, question

        return highest_valued_gain, highest_valued_question

class Leaf:
    def __init__(self, rows):
        self.predictions = self.classCount(rows)
    def classCount(self,rows):
        counts = {}
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        value = example[self.column]
        if self.isNum(value):
            return value >= self.value
        else:
            return value == self.value

    def isNum(self,value):
        return isinstance(value, int) or isinstance(value, float)

