import numpy as np

y = np.load('y.npy')

print(y.shape[0])
print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))
