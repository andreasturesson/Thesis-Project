import randomForest as rf
'''
Code is inspired or acquired from: Sebastian Mantey(youtube) and AI Sciences, AI Sciences Team(udemy)
Source: https://www.youtube.com/watch?v=WvmPnGmCaIM&list=PLPOTBrypY74y0DviMOagKRUhDdk0JyM_r (Obtained: 2021/02/01)
Source: https://www.udemy.com/course/decision-tree-and-random-forest-python-from-zero-to-hero/ (Obtained: 2021/01/20)
'''

if __name__ == '__main__':
    databas = "../../../../Dataset/dataset_ensemble/4.0/dataset_4.1_sliced.csv"
    #rfa0 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=1, MAX_DEPTH= 50)
    rfa1 = rf.RandomForest(DATASET=databas, FOREST_SIZE=10, MAX_DEPTH=500)
    '''
    rfa2 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=10, MAX_DEPTH= 50)
    rfa3 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=25, MAX_DEPTH= 50)
    rfa4 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=25, MAX_DEPTH= 50)

    rfa0 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=1, MAX_DEPTH= 10)
    rfa1 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=5, MAX_DEPTH= 25)
    rfa2 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=10, MAX_DEPTH= 50)
    rfa3 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=25, MAX_DEPTH= 75)
    rfa4 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=25, MAX_DEPTH= 100)

    rfa4 = rf.RandomForest(DATASET="dataset_3.1.csv", FOREST_SIZE=25, MAX_DEPTH=-1)
    '''