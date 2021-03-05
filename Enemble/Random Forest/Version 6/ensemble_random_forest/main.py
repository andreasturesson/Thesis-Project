import pandas as pd
import numpy as np
import warnings
import tqdm
from openpyxl.workbook import Workbook
from IPython.display import Image
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, plot_confusion_matrix

np.set_printoptions(formatter={'float:kind': '{:f}'.format})
sns.set(rc={'figure.figsize': (8, 6)})

from pandas import to_datetime
import itertools
import os
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import datetime

warnings.filterwarnings('ignore')

PATH = 'C:\\Users\\Jonathan\\Thesis-Project\Enemble\\Random Forest\\Version 6\\ensemble_random_forest\\pic\\'
Image(filename=PATH + "Machine Learning.png", width=900, height=900)


def prepareDataset():
    df = pd.read_csv('../../../../Dataset/dataset_ensemble/5.0/dataset_5.4.csv', low_memory=False)
    labels = np.array(df['label'])
    dataset = df.drop('label', axis=1)  # Saving feature names for later use
    feature_list = list(dataset.columns)  # Convert to numpy array
    dataset = np.array(dataset)
    return train_test_split(dataset, labels, test_size=0.2, random_state=420)

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size': 50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

def printShape(X_train, X_test, Y_train, Y_test):
    print('X_train Shape:', X_train.shape)
    print('X_test Shape:', X_test.shape)
    print('Y_train Shape:', Y_train.shape)
    print('Y_test Shape:', Y_test.shape)

def metricData(cm):
    # Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    TN = cm[0][0]

    # Sensitivity, hit rate, recall, or true positive rate
    recall = TP / (TP + FN)
    print("The Recall per class is: ", recall)

    # Precision or positive predictive value
    precision = TP / (TP + FP)
    print("The Precision per class is: ", precision)

    # F1 score
    F1_score = 2*((precision * recall) / (precision + recall))
    print("The F1_score per class is: ", F1_score)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("The Accuracy of each class is", ACC)
    print("")

def plot_confusion_matrix(rfc):

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cm_norm)
    plt.figure()
    plot_confusion_matrix(cm_norm, classes=model.classes_)
    pass

def VideoGuide():
    #model.fit(X_train, Y_train)
    #prediction_test = model.predict(X=X_test)
    # print('Classification accuracy on test set with trees = {}, max features = {} and max_depth = {}: {:.3f}'.format(n_e, m_f, m_d,accuracy_score(Y_test, prediction_test)))
    #cm = confusion_matrix(Y_test, prediction_test)
    #cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    # plt.figure()
    # plot_confusion_matrix(cm_norm, classes=model.classes_, title='Confusion matrix accuracy on test set with trees = {}, max features = {} and max_depth = {}: {:.3f}'.format(n_e, m_f, m_d, accuracy_score(Y_test, prediction_test)))
    pass

def plot():
    # plot_precision = []
    # plot_recall = []
    # plot_F1_score = []
    # plot_ACC = []
    # x_estimators = np.array([])
    #
    # n_estimators = [10, 25, 50, 75, 100]
    # max_features = 'sqrt'
    # max_depths = [25, 50, 75, 100, None]
    # for n_e, m_d in product(n_estimators, max_depths):
    #     model = RandomForestClassifier(n_estimators=n_e, criterion='gini', max_features='sqrt', max_depth=m_d
    #                                    , n_jobs=5, random_state=420)
    #     model.fit(X_train, Y_train)
    #     prediction_test = model.predict(X=X_test)
    #     cm = confusion_matrix(Y_test, prediction_test)
    #     TP = cm[1][1]
    #     FN = cm[1][0]
    #     FP = cm[0][1]
    #     TN = cm[0][0]
    #     plot_precision.append(TP / (TP + FP))
    #     plot_recall.append(TP / (TP + FN))
    #     plot_F1_score.append(2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN)))))
    #     plot_ACC.append((TP + TN) / (TP + FP + FN + TN))
    #     x_estimators = np.append(x_estimators, n_e)
    #     print(n_e)
    #
    # plt.plot(x_estimators, plot_precision)
    # plt.plot(x_estimators, plot_recall)
    # plt.plot(x_estimators, plot_F1_score)
    # plt.plot(x_estimators, plot_ACC)
    # plt.xticks(np.arange(min(x_estimators)-10, max(x_estimators) + 100, 100.0))
    #
    # plt.legend(['Prec', 'Recall', 'F1', 'ACC'], loc='upper left')
    # plt.show()
    pass

def writeResultToExcel():
        X_train, X_test, Y_train, Y_test = prepareDataset()

        randomForest_Result = pd.DataFrame({'Trees': [],
                                            'Depth': [],
                                            'Features': [],
                                            'Random_state': [],
                                            'ACC': [],
                                            'F1_score': [],
                                            'Precision': [],
                                            'Recall': []
                                            })
        n_estimators = []
        max_depths = [30, 60, 90]
        for y in range(800, 1075, 50):
            n_estimators.append(y)
        # for x in range(25, 500, 25):
        #    max_depths.append(x)
        max_depths.append(None)
        # log2=6, sqrt= 7, 10, 13 , 47/3=16, 20,
        max_features = [13, 15, 18, 21]
        random_state = [1337, 4553, 412]
        temp = 0
        for n_e, m_d, m_f, r_s in product(n_estimators, max_depths, max_features, random_state):
            model = RandomForestClassifier(n_estimators=n_e, criterion='gini', max_features=m_f, max_depth=m_d
                                           , n_jobs=-1, random_state=r_s)
            model.fit(X_train, Y_train)
            prediction_test = model.predict(X=X_test)
            cm = confusion_matrix(Y_test, prediction_test)
            TP = cm[1][1]
            FN = cm[1][0]
            FP = cm[0][1]
            TN = cm[0][0]
            temp_df = pd.DataFrame({'Trees': [n_e],
                                    'Depth': [m_d],
                                    'Features': [m_f],
                                    'Random_state': [r_s],
                                    'ACC': [((TP + TN) / (TP + FP + FN + TN))],
                                    'F1_score': [(2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / (
                                            (TP / (TP + FP)) + (TP / (TP + FN)))))],
                                    'Precision': [(TP / (TP + FP))],
                                    'Recall': [(TP / (TP + FN))]
                                    })
            randomForest_Result = randomForest_Result.append(temp_df, ignore_index=True)

            if temp != n_e:
                print(n_e)
            temp = n_e

        randomForest_Result.to_excel('pic/trial2.xlsx', index=False, header=True)

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = prepareDataset()
