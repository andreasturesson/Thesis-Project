training_result

Each gridsearch use specific hyperparameters and dataset(orderd by versions).

gridsearch_*_value_original: The original file without any changes.

gridsearch_*_value: Contains results from a the best results. Using a fraction of the best results
to create diagrams to patters.

gridsearch_*_without_mean_original: The original file without any changes. Mean indicates more than
one random seed was used.

gridsearch_*_mean_value_original: Processed excel file combining the random seed from "gridsearch_*_without_mean_original"
to get a mean value from ACC, F1 score, Precision and Recall. 

gridsearch_*_mean_value: Contains results from a the best results. Using a fraction of the best results
to create diagrams to patters. Based on "gridsearch_*_mean_value_original".

----------------------------------------------------------------------------------------------
Gridsearch one

Dataset: Version 6.5 dataset 'train_dataset' with 47 features

Hyperparameters:

n_estimators: 	10, 25, 50, 75...975
max_features: 	10, 13, 16, ‘sqrt’ and ‘log2’
max_depth: 	10, 25, 50, 75...500 and ‘None’
random_state: 	1337
----------------------------------------------------------------------------------------------
Gridsearch two

Dataset: Version 6.5 dataset 'train_dataset' with 47 features

Hyperparameters:

n_estimators: 	800, 850,900, 950, 1000 and 1050
max_features:	13, 15, 18 and 21
max_depth:	30, 60, 90 and ‘None’
random_state: 	1337, 4553 and 412
----------------------------------------------------------------------------------------------
Gridsearch three

Dataset: Version 7.5 dataset 'train_dataset' with 48 features(updated features)

Hyperparameters:

n_estimators: 	800, 850,900, 950, 1000 and 1050
max_features:	13, 15, 18 and 21
max_depth:	30, 60, 90 and ‘None’
random_state: 	1337, 4553 and 412
----------------------------------------------------------------------------------------------

Test_result

Contain an excel file with the final precition and the actual predicitons on the hold-out validaition

This is used for McNemar Test to find out if there is a significant difference between 2 classifiers.

