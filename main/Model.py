import pitt.perceptron as pitt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    s = 'https://archive.ics.uci.edu/ml/' \
        'machine-learning-databases/iris/iris.data'
    print('From URL', s)
    df = pd.read_csv(s, header=None, encoding='utf-8')
    df.tail()
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend('upper left')
    plt.show()

    

main()
