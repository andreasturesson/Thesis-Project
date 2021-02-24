import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dnsrequest_dataset = pd.read_csv('dataset_3.1.csv')

def randonForest():

    X = dnsrequest_dataset.drop('label', axis=1)
    Y = dnsrequest_dataset['label']
    X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size= 0.2)

    rfc = RandomForestClassifier(n_estimators= 10)

    rfc.fit(X_training, Y_training)
    y_predict = rfc.predict(X_test)
    print(accuracy_score(Y_test, y_predict)*100)
    return rfc


def featureImportance(rfc):
    cols = dnsrequest_dataset.columns.drop('label')
    feature_importances = pd.Series(rfc.feature_importances_, index = cols).sort_values(ascending=False)
    print(feature_importances)
    feature_importances.to_csv("feature_importance.csv", header=True, mode='a')

if __name__ == '__main__':
    trainedRFC1 = randonForest()
    featureImportance(trainedRFC1)

    trainedRFC2 = randonForest()
    featureImportance(trainedRFC2)

    trainedRFC3 = randonForest()
    featureImportance(trainedRFC3)
    df = pd.read_csv("feature_importance.csv", names=["feature", "value"])
    df = df.groupby(['feature'])['value'].agg(lambda x: x.unique().sum()/x.nunique())
    print(df)

    os.remove('feature_importance.csv')

    #df.to_csv("avg.csv", float_format='%.15f', mode='a')



