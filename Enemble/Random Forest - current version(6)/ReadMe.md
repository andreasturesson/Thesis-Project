# Random Forest Classifier(RF) - version 6

## Docummentation:

Version 6 utilize sklearn library compared to previous.

This class run separate functions after the users need, the user is responsible for
changing paths, file names and hyper-parameters.

## Functions:

Note: Changing hyper-parameters are done manually inside functions.
      Changing paths are done manually inside functions.
      Changing the data set in manually done in prepareDataset()

### def prepareDataset()
    General function for loading a dataset from a csv file. Splits the data in 4
    numpy arrays. x_test, x_train, Y_test and Y_train.
    Used for training the RF model. Default settings are a 80/20 split with 420 as random seed

### def prepareFinalTrainData()
    Modified version of prepareDataset(), separates the lable form the data without
    splitting or shuffling.
    Returns X_train and Y_train (all of the loaded data)

### def prepareFinalTestData()
    Modified version of prepareDataset(), separates the lable form the data without
    splitting or shuffling.
    Returns X_test and Y_test (all of the loaded data)

### def printShape(X_train, X_test, Y_train, Y_test)
    return the shapes of numpy array recived from prepareDataset()

### def writeResultToExcel():
    Grid search function for random forest. Prepares a dataframe containing hyper-parameters
    and metrics. The function will run all combination of hyper-parameters and when finished
    write the results to an excel file.

### def multi_classification():
    Experimental function to evaluate how the different DGA-families was classified.

### def multi_clssification_misslcassified():
    Measures only what classes was miscalculated in terms of DGA-domain and legitimate domain,
    Even if a DGA-domain from 'kraken_v2' was classified as 'chinad', it does not count as
    misclassified as it still classified it as an DGA-domain.

### def writeResultToConsole():
    Same as  writeResultToExcel(), the difference is that is writes the results directly
    to the console instead of saving the data inside of an dataframe.

### def getMeanFromDifferentRandomSeeds():
    Reads an excel file from writeResultToExcel() containing three random seeds and calculates
    the mean value.
    Returns an excel file identical to writeResultToExcel() with the mean value from different seeds,

### def finalResult():
    Creates a single random forest classifier with set hyper-parameters and uses the modified
    perpareDataset() function. Writes the predicted values and actual values to an excel document
    to later evaluate McNemars test.
    Prints hyper-paramets, confusion metric, ACC, F1 score, Precision and Recall to the consol.
