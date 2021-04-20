import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report
# from header.dataset import load_data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from header.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # Test random forest
    dataset = load_digits()
    X, Y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    # x_train, y_train, x_test = load_data('./data/x_train.csv', './data/y_train.csv', './data/x_test.csv')
    # print(x_train.shape, y_train.shape, x_test.shape)
    clf = RandomForestClassifier(
        n_estimators = 40, max_depth =20, max_features = 'sqrt',
        min_impurity_decrease = 0.08, verbose = 1, class_weight = 'balanced',
        min_samples_split = 8, max_samples = 0.75, n_jobs = 8
    )

    # clf = DecisionTreeClassifier(
    #     max_depth = 30, max_features = 'sqrt', min_impurity_decrease = 0.08,
    #     class_weight = 'balanced'
    # )

    print('Start to train...')
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()

    print(f'End in {(end - start):.3f}s.\n')
    predict_result = clf.predict(x_test)
    print(classification_report(y_train, clf.predict(x_train)))

    
    
    # output predict result
    # with open('./data/ans.txt', 'w') as outfile:
    #     for result in predict_result:
    #         outfile.write(f"{result}\n")
    #         # print(result)

    
