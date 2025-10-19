import numpy as np
import pandas

def leastSquares(x, y, deg=1):
    # Матрица Вандермонда
    a = np.vander(x, deg + 1)
    # Матрица Пенроуза
    pinv_a = np.linalg.pinv(a)
    w = pinv_a.dot(y)
    return w

df = pandas.read_csv("/home/dmitry/Data/flow/example-01.csv")

x_ser = df.x.tolist()
y_ser = df.y.tolist()