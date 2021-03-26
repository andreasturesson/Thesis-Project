import pandas as pd
import numpy as np
import warnings
from IPython.display import Image
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, \
    confusion_matrix, plot_confusion_matrix, classification_report, ConfusionMatrixDisplay

np.set_printoptions(formatter={'float:kind': '{:f}'.format})
sns.set(rc={'figure.figsize': (8, 6)})

warnings.filterwarnings('ignore')

PATH = 'C:\\Users\\Jonathan\\Thesis-Project\Enemble\\Random Forest\\Version 6\\ensemble_random_forest\\pic\\'

class_names = ["Alexa_Majestic",
                                "kraken_v2",
                                "ramnit",
                                "vawtrak_v2",
                                "murofet_v3",
                                "chinad",
                                "ccleaner",
                                "gozi_nasa",
                                "zeus-newgoz",
                                "locky",
                                "proslikefan",
                                "corebot",
                                "kraken_v1",
                                "cryptolocker",
                                "banjori",
                                "nymaim",
                                "shiotob",
                                "ramdo",
                                "murofet_v1",
                                "ranbyus_v1",
                                "dircrypt",
                                "suppobox_2",
                                "necurs",
                                "suppobox_3 ",
                                "vawtrak_v1 ",
                                "alureon ",
                                "gozi_luther ",
                                "matsnu ",
                                "pykspa_noise",
                                "pushdo",
                                "qakbot",
                                "fobber_v2",
                                "vawtrak_v3",
                                "symmi",
                                "pykspa",
                                "gozi_gpl",
                                "pizd",
                                "dyre",
                                "bedep",
                                "sisron",
                                "qadars",
                                "tempedreve",
                                "gozi_rfc4343",
                                "simda",
                                "ranbyus_v2",
                                "fobber_v1",
                                "rovnix",
                                "suppobox_1",
                                "padcrypt",
                                "murofet_v2",
                                "tinba"]

def prepareDataset():
    df = pd.read_csv('../../../../Dataset/dataset_ensemble/7.0/7.1/train_dataset.csv', low_memory=False)
    labels = np.array(df['label'])
    dataset = df.drop('label', axis=1)  # Saving feature names for later use
    feature_list = list(dataset.columns)  # Convert to numpy array
    dataset = np.array(dataset)
    return train_test_split(dataset, labels, test_size=0.2, random_state=420)

#  X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=1, random_state=420)
def prepareFinalTrainData():
    df = pd.read_csv('../../../../Dataset/dataset_ensemble/7.0/7.1/train_dataset.csv', low_memory=False)
    Y_train = np.array(df['label'])
    dataset = df.drop('label', axis=1)  # Saving feature names for later use
    X_train = np.array(dataset)
    print(X_train)
    print(Y_train)
    return X_train, Y_train

def prepareFinalTestData():
    df = pd.read_csv('../../../../Dataset/dataset_ensemble/7.0/7.1/test_dataset.csv', low_memory=False)
    Y_test = np.array(df['label'])
    dataset = df.drop('label', axis=1)  # Saving feature names for later use
    X_test = np.array(dataset)
    print(X_test)
    print(Y_test)
    return X_test, Y_test

def printShape(X_train, X_test, Y_train, Y_test):
    print('X_train Shape:', X_train.shape)
    print('X_test Shape:', X_test.shape)
    print('Y_train Shape:', Y_train.shape)
    print('Y_test Shape:', Y_test.shape)

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
    max_depths = [30, 60, 90, 110]
    for y in range(800, 1001, 100):
        n_estimators.append(y)
    # for x in range(25, 500, 25):
    #    max_depths.append(x)
    max_depths.append(None)
    # log2=6, sqrt= 7, 10, 13 , 47/3=16, 20,
    max_features = ['log2', 'sqrt', 9, 13, 15, 18]
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

    randomForest_Result.to_excel('pic/trial3.xlsx', index=False, header=True)

def multi_classification():
    X_train, X_test, Y_train, Y_test = prepareDataset()

    model = RandomForestClassifier(n_estimators=10, criterion='gini', max_features=14, max_depth=125
                                   , n_jobs=-1, random_state=1337)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X_test, Y_test,
                                     display_labels=class_names,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        # print(title)
        # print(disp.confusion_matrix)

    plt.show()

    print('\nClassification Report\n')
    print(classification_report(Y_test, Y_pred, target_names=class_names))

def multi_clssification_misslcassified():
    # Multi-classification on what families was believed to be a real domain.
    X_train, X_test, Y_train, Y_test = prepareDataset()
    model = RandomForestClassifier(n_estimators=10, criterion='gini', max_features=14, max_depth=125
                                   , n_jobs=-1, random_state=1337)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    missclassified = np.zeros((len(class_names), 2), dtype = object)
    for pred, actual in zip(Y_pred, Y_test):
        if pred == 0 and actual != 0:
            missclassified[actual][0] = missclassified[actual][0] + 1
    missclassified_normalized = missclassified
    missclassified_normalized.astype(dtype=float)

    sum = np.sum(missclassified)
    for idx, (a, _) in enumerate(missclassified_normalized):
        missclassified_normalized[idx][0] = a / sum
        missclassified_normalized[idx][1] = class_names[idx]

    # bror fråga stackoverflow nissarna
    missclassified_normalized = missclassified_normalized[missclassified_normalized[:, 0].argsort()[::-1]]
    print(missclassified_normalized)

def writeResultToConsole():
    X_train, X_test, Y_train, Y_test = prepareDataset()


    # log2=6, sqrt= 7, 10, 13 , 47/3=16, 20,
    max_features = [17, 20]
    random_state = [1337]
    n_estimators = [500]
    max_depths = [30, 60]

    for n_e, m_d, m_f, r_s in product(n_estimators, max_depths, max_features, random_state):
        model = RandomForestClassifier(n_estimators=n_e, criterion='gini', max_features=m_f, max_depth=m_d
                                       , n_jobs=-1)
        model.fit(X_train, Y_train)
        prediction_test = model.predict(X=X_test)
        cm = confusion_matrix(Y_test, prediction_test)
        TP = cm[1][1]
        FN = cm[1][0]
        FP = cm[0][1]
        TN = cm[0][0]

        print('Trees: ', n_e)
        print('Depth: ', m_d)
        print('Features: ', m_f)
        print('Random_state: ', r_s)
        print('ACC: ', ((TP + TN) / (TP + FP + FN + TN)))
        print('F1_score: ', (2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))))
        print('Precision: ', (TP / (TP + FP)))
        print('Recall: ', (TP / (TP + FN)))
        print('----------------------------------------------------------------------------------------')

def getMeanFromDifferentRandomSeeds():
    # Excel sheet with different random seeds
    excel_sheet_with_random_seed = pd.read_excel('pic/trial3.xlsx')
    # Arrays to find all unique values for tree, depth and features
    unique_trees = np.array([])
    unique_depths = np.array([])
    unique_features = np.array([])
    # New skeleton for the excel_mean
    excel_sheet_with_mean = pd.DataFrame({'Trees': [],
                                          'Depth': [],
                                          'Features': [],
                                          'ACC_mean': [],
                                          'F1_score_mean': [],
                                          'Precision_mean': [],
                                          'Recall_mean': []
                                          })

    # Find unique values for Trees, Depth, and features
    for index, row in excel_sheet_with_random_seed.iterrows():
        if row['Trees'] not in unique_trees:
            unique_trees = np.append(unique_trees, row['Trees'])
        if row['Depth'] not in unique_depths:
            unique_depths = np.append(unique_depths, row['Depth'])
        if row['Features'] not in unique_features:
            unique_features = np.append(unique_features, row['Features'])

    print(unique_trees)
    print(unique_depths)
    print(unique_features)

    # Grid search: Looking at the unique values and if they match it will summarize ACC, F1_score, Recall and Precision
    # Then take the summarized value and divide it with the number of rows found to get the mean value for that
    # specific combination.
    for u_tree, u_depth, u_feature in product(unique_trees, unique_depths, unique_features):
        sum_ACC, sum_F1_score, sum_precision, sum_recall = np.array([]), np.array([]), np.array([]), np.array([])
        for index, row in excel_sheet_with_random_seed.iterrows():
            if row['Trees'] == u_tree and row['Depth'] == u_depth and row['Features'] == u_feature:
                sum_ACC = np.append(sum_ACC, row['ACC'])
                sum_F1_score = np.append(sum_F1_score, row['F1_score'])
                sum_precision = np.append(sum_precision, row['Precision'])
                sum_recall = np.append(sum_recall, row['Recall'])

        temp_df = pd.DataFrame({'Trees': [u_tree],
                                'Depth': [u_depth],
                                'Features': [u_feature],
                                'ACC_mean': [np.sum(sum_ACC) / sum_ACC.size],
                                'F1_score_mean': [np.sum(sum_F1_score) / sum_F1_score.size],
                                'Precision_mean': [np.sum(sum_precision) / sum_precision.size],
                                'Recall_mean': [np.sum(sum_recall) / sum_recall.size]
                                })

        excel_sheet_with_mean = excel_sheet_with_mean.append(temp_df, ignore_index=True)

    excel_sheet_with_mean.to_excel('pic/trial3_mean_value.xlsx', index=False, header=True)

def finalResult():
    # Dataset 7.1
    X_train, Y_train = prepareFinalTrainData()
    X_test, Y_test = prepareFinalTestData()

    model = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features=18, max_depth=40
                                   , n_jobs=-1, random_state=13)
    model.fit(X_train, Y_train)
    prediction_test = model.predict(X=X_test)
    mcNemarsTest = pd.DataFrame({'actual':Y_test, 'predicted': prediction_test}, columns=['actual', 'predicted'])
    mcNemarsTest.to_excel('test_result/mcNemars_final_result.xlsx', index=False, header=True)

    cm = confusion_matrix(Y_test, prediction_test)
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    TN = cm[0][0]

    print('----------------------------------------------------------------------------------------')
    print('Trees: ', 1000)
    print('Depth: ', 40)
    print('Features: ', 18)
    print('Random_state: ', 13)
    print('ACC: ', ((TP + TN) / (TP + FP + FN + TN)))
    print('F1_score: ', (2 * (((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))))
    print('Precision: ', (TP / (TP + FP)))
    print('Recall: ', (TP / (TP + FN)))
    print('TP: %s  FN: %s  FP: %s  TN: %s' % (TP, FN, FP, TN))
    print('----------------------------------------------------------------------------------------')

if __name__ == '__main__':
    finalResult()