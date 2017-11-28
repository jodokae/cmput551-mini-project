import numpy as np
from scipy import stats
from prettytable import PrettyTable
from sklearn.metrics import auc

confusion_matrix = np.load('confusion.npy')
  
# tpr = sensitivity
tpr = confusion_matrix[:,:,0] / (confusion_matrix[:,:,0] + confusion_matrix[:,:,2])
# tnr = specifity
tnr = confusion_matrix[:,:,3] / (confusion_matrix[:,:,1] + confusion_matrix[:,:,3])

total = confusion_matrix[:,:,0] + confusion_matrix[:,:,1] + confusion_matrix[:,:,2] + confusion_matrix[:,:,3]
correctLabeled = confusion_matrix[:,:,0] + confusion_matrix[:,:,3]
acc = correctLabeled / total


data = np.empty((3,acc.shape[0], acc.shape[1]))
data[0] = acc
data[1] = tpr
data[2] = tnr
label = ['Accurarcy', 'Sensitivity', 'Specificity']
algorithms = ['Mean', 'Bayes', 'Tree', 'Neural Net']

for metric in range(3):
    print(label[metric])
    x = PrettyTable()
    y = PrettyTable()
    x.field_names = [''] + algorithms
    y.field_names = [''] + algorithms
    for i in range(4):
        row = [algorithms[i], '', '', '', '']
        yrow = [algorithms[i], '', '', '', '']
        for j in range(4):
            #print(algorithms[i])
            #print(algorithms[j])
            (t, p) = stats.ttest_ind(a=data[metric][i], b = data[metric][j], equal_var=False)
            row[j+1] = format(t, '.4g') + ' ' + format(p, '.3g')
            
            if p > 0.05 or np.isnan(p):
                v = 'same'
            elif t > 0:
                v = 'better'
            else:
                v = 'worse'
            yrow[j+1] = v
        x.add_row(row)
        y.add_row(yrow)
    print(x)
    print(y)

