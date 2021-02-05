import randomForest as rf
'''
Code is inspired or acquired from: Sebastian Mantey(youtube) and AI Sciences, AI Sciences Team(udemy)
Source: https://www.youtube.com/watch?v=WvmPnGmCaIM&list=PLPOTBrypY74y0DviMOagKRUhDdk0JyM_r (Obtained: 2021/02/01)
Source: https://www.udemy.com/course/decision-tree-and-random-forest-python-from-zero-to-hero/ (Obtained: 2021/01/20)
'''

if __name__ == '__main__':
    dataset = "../../../../Dataset/dataset_ensemble/4.0/dataset_4.1.csv"
    rfa4 = rf.RandomForest(DATASET=dataset, FOREST_SIZE=100, MAX_DEPTH= 100, EPOCH=3, RANDOM_SUBSPACE=10) # 30 h



