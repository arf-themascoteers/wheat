import numpy as np
import matplotlib.pyplot as plt
from ds_manager import DSManager
import utils

dm = DSManager()

data = dm.data

data = data[:,[12,13]]

sorted_indices = np.argsort(data[:, 0])
data = data[sorted_indices]

n = data[:,0]
savi = data[:,1]
plt.plot(n,savi)
plt.show()

pc = utils.calculate_pc(n, savi)
print(pc)

