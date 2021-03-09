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
Image(filename=PATH + "Machine Learning.png", width=900, height=900)
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
    df = pd.read_csv('../../../../Dataset/dataset_ensemble/6.0/6.5/train_multiclass.csv', low_memory=False)
    labels = np.array(df['label'])
    dataset = df.drop('label', axis=1)  # Saving feature names for later use
    feature_list = list(dataset.columns)  # Convert to numpy array
    dataset = np.array(dataset)
    return train_test_split(dataset, labels, test_size=0.2, random_state=420)

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
    F1_score = 2 * ((precision * recall) / (precision + recall))
    print("The F1_score per class is: ", F1_score)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("The Accuracy of each class is", ACC)
    print("")

def VideoGuide():
    # model.fit(X_train, Y_train)
    # prediction_test = model.predict(X=X_test)
    # print('Classification accuracy on test set with trees = {}, max features = {} and max_depth = {}: {:.3f}'.format(n_e, m_f, m_d,accuracy_score(Y_test, prediction_test)))
    # cm = confusion_matrix(Y_test, prediction_test)
    # cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    # plt.figure()
    # plot_confusion_matrix(cm_norm, classes=model.classes_, title='Confusion matrix accuracy on test set with trees = {}, max features = {} and max_depth = {}: {:.3f}'.format(n_e, m_f, m_d, accuracy_score(Y_test, prediction_test)))
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

if __name__ == '__main__':
    pass # the beer
