import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dnsrequest_dataset = pd.read_csv('dataset.csv')

def randonForest():

    X = dnsrequest_dataset.drop('labeled', axis=1)
    Y = dnsrequest_dataset['labeled']
    X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size= 0.2)

    rfc = RandomForestClassifier(n_estimators= 15)

    rfc.fit(X_training, Y_training)
    y_predict = rfc.predict(X_test)
    print(accuracy_score(Y_test, y_predict)*100)
    return rfc


def featureImportance(rfc):
    cols = dnsrequest_dataset.columns.drop('labeled')
    feature_importances = pd.Series(rfc.feature_importances_, index = cols).sort_values(ascending=False)
    print(feature_importances)

if __name__ == '__main__':
    trainedRFC1 = randonForest()
    featureImportance(trainedRFC1)

    trainedRFC2 = randonForest()
    featureImportance(trainedRFC2)

    trainedRFC3 = randonForest()
    featureImportance(trainedRFC3)



