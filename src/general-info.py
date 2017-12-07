import numpy as np

y = np.load('y.npy')

print(y.shape[0])
print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))

A = np.load('A.npy')

missing = (np.count_nonzero(np.isnan(A), axis=0) / y.shape[0])
for i in range(missing.shape[0]):
    print(str(i) + ': ' + str(missing[i]))
