import numpy as np
import commondef
import matplotlib.pyplot as plt

wm = commondef.leastSquares(commondef.x_ser, commondef.y_ser,1)

z_ser = np.polyval(wm, commondef.x_ser)

plt.plot(commondef.x_ser, commondef.y_ser)
plt.plot(commondef.x_ser, z_ser)
plt.show()
