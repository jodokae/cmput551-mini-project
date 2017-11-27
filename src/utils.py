import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable 


def printMeanedMatrix(matrix):
    means = np.mean(matrix, axis=0)
    var = np.var(matrix, axis=0)
    printMatrix(means[0], means[1], means[2], means[3], 
        var[0], var[1], var[2], var[3])

def printMatrix(tp, fp, fn, tn, tp_v, fp_v, fn_v, tn_v):
    total = tp + fp + fn + tn
    correctLabeled = tp + tn
    acc = correctLabeled / total
    
    print("Number of mislabeled points out of a total %d points : %d" % (total,(total-correctLabeled)))
    
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",meanVarToString(tp, tp_v), meanVarToString(fp, fp_v), meanVarToString(tp+fp, tp_v+fp_v)])
    x.add_row(["False (Pred)",meanVarToString(fn, fn_v), meanVarToString(tn, tn_v), meanVarToString(fn+tn, fn_v+tn_v)])
    x.add_row(["Sum", meanVarToString(tp+fn, 0), meanVarToString(fp + tn, 0), meanVarToString(total, 0)])
    print(x)
    
    print('Acc: ' + str(acc))
    
    
def meanVarToString(mean, var):
    return str(mean) + ' +- ' + str(round(np.sqrt(var), 2))
    
def load(full):
    print ('Loading Data')
    if full==True:
        A = np.load('A.npy')
        y = np.load('y.npy')
    else :
        A = np.load('A_small.npy')
        y = np.load('y_small.npy')

    print ('Encoding Data')
    
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(A)

    X = imp.transform(A)
    
    return (X, y)
    
def loadAndSplit(fullSet):

    X, y = load(fullSet)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
    return (X_train, X_test, y_train, y_test)
    
def splitData(X, y, numruns):
    print ('Splitting Data')

    sss = StratifiedShuffleSplit(n_splits=numruns, test_size=0.2)

    X_learn = []
    X_test = []
    y_learn = []
    y_test = []

    for train_index, test_index in sss.split(X, y):
        X_learn.append(X[train_index])
        X_test.append(X[test_index])
        y_learn.append(y[train_index])
        y_test.append(y[test_index])

    return (X_learn, X_test, y_learn, y_test)
