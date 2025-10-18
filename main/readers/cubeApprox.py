import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv("/home/dmitry/Data/flow/example-01.csv")

x_ser = df.x.tolist()
y_ser = df.y.tolist()

def leastSquares(x, y, deg=1):
    a = np.vander(x, deg + 1)
    pinv_a = np.linalg.pinv(a)
    w = pinv_a.dot(y)
    return w

wm = leastSquares(x_ser, y_ser,3)

z_ser = np.polyval(wm, x_ser)

plt.plot(x_ser, y_ser)
plt.plot(x_ser, z_ser)
plt.show()
