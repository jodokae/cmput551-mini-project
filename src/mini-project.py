import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score   
import itertools

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def doStuff(learner, X_learn, y_learn, X_test, y_test):
    learner.fit(X_learn, y_learn)

    y_pred = learner.predict(X_test)
    
    #printRes(y_test, y_pred)
    
    return accuracy_score(y_test, y_pred)
    
    
def printRes(y_test, y_pred):
    print("Number of mislabeled points out of a total %d points : %d" % (y_test.shape[0],(y_test != y_pred).sum()))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
    print(x)
    
    print('Acc: ' + str(accuracy_score(y_test, y_pred)))


print ('Loading Data')

A = np.load('A_small.npy')
y = np.load('y_small.npy')

print ('Encoding Data')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(A)

X = imp.transform(A)

print ('Splitting Data')

numruns = 2

sss = StratifiedShuffleSplit(n_splits=numruns, test_size=0.1, random_state=0)

X_learn = []
X_test = []
y_learn = []
y_test = []

for train_index, test_index in sss.split(X, y):
    X_learn.append(X[train_index])
    X_test.append(X[test_index])
    y_learn.append(y[train_index])
    y_test.append(y[test_index])


treeParams = list(dict_product(dict(
    criterion=['gini'],
    splitter=['random'],
    random_state=[None],
    max_depth=[1, 10, 100], 
    #min_samples_split=[1e-5, 1e-4, 1e-3], 
    #min_samples_leaf=[1e-5, 1e-4, 1e-3],
    #min_weight_fraction_leaf=[1e-5, 1e-4, 1e-3],
    max_features=['sqrt', 'log2'],
    #max_leaf_nodes=[100, 1000, 10000]
)))
nnParams = list(dict_product(dict(
    activation=['logistic'],
    hidden_layer_sizes=[(10,), (50,), (100,)],
    #alpha=[1e-4, 1e-3, 1e-2],
    #learning_rate_init=[1e-3, 1e-2, 1e-1],
    #beta_1=[0.1, 0.5, 0.9], 
    #beta_2=[0.1, 0.5, 0.9]
))) 


numMeanParams = 1
numBayesParams = 1
numTreeParams = len(treeParams)
numNnParams = len(nnParams)

meanAcc = np.zeros((numruns, numMeanParams))
bayesAcc = np.zeros((numruns, numBayesParams))
treeAcc = np.zeros((numruns, numTreeParams))
nnAcc = np.zeros((numruns, numNnParams))

print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))

for split in range(numruns):

    print('Run ' + str(split) + '/' + str(numruns))
    
    
    #print('\n Mean \n\n')

    pred = 0 if ((y_learn[split] == 0).sum()) > ((y_learn[split] ==1).sum()) else 1
    y_pred = np.zeros(y_test[split].shape) + pred
    
    meanAcc[split][0] = accuracy_score(y_test[split], y_pred) 
    
    #printRes(y_test[split], y_pred)
    
        
    #print('\n Gaussian Naive Bayes \n\n')

    gnb = GaussianNB()
    bayesAcc[split][0] = doStuff(gnb, X_learn[split], y_learn[split], X_test[split], y_test[split])

    
    
    
    #print('\n Decision Tree Gini \n\n')   
    
    for index, params in enumerate(treeParams):
        clf = DecisionTreeClassifier(**params)
    
        #print(clf.get_params())
   
        acc = doStuff(clf, X_learn[split], y_learn[split], X_test[split], y_test[split])
        
        print(str(index+1) + '/' + str(numTreeParams) + ': ' + str(acc))
        
        treeAcc[split][index] = acc
        
    #print(clf.decision_path(X_test[split][0].reshape((1, X_test[split].shape[1]))))
    
        
    
    #print('\n Neural Net \n\n')
    
    for index, params in enumerate(nnParams):
        clf = MLPClassifier(**params)
    
        #print(clf.get_params())
   
        acc = doStuff(clf, X_learn[split], y_learn[split], X_test[split], y_test[split])
        
        print(str(index+1) + '/' + str(numNnParams) + ': ' + str(acc))
        
        nnAcc[split][index] = acc
    

meanTreeAcc = np.mean(treeAcc, axis=0)

bestTreeAcc = 0
bestTreeFeat = {}
for i, mean in enumerate(meanTreeAcc):
    if(mean > bestTreeAcc):
        bestTreeAcc = mean
        bestTreeFeat = treeParams[i]

meanNnAcc = np.mean(nnAcc, axis=0)

bestNnAcc = 0
bestNnFeat = {}
for i, mean in enumerate(meanNnAcc):
    if(mean > bestNnAcc):
        bestNnAcc = mean
        bestNnFeat = nnParams[i]
        
meanMeanAcc = np.mean(meanAcc, axis=0)

bestMeanAcc = 0
bestMeanFeat = {}
for i, mean in enumerate(meanMeanAcc):
    if(mean > bestMeanAcc):
        bestMeanAcc = mean
        
meanBayesAcc = np.mean(bayesAcc, axis=0)

bestBayesAcc = 0
bestBayesFeat = {}
for i, mean in enumerate(meanBayesAcc):
    if(mean > bestBayesAcc):
        bestBayesAcc = mean

print('Mean: Acc ' + str(bestMeanAcc))
print('Bayes: Acc ' + str(bestBayesAcc))
print('Tree: Acc ' + str(bestTreeAcc) + ' : ' + str(bestTreeFeat))
print('NN: Acc ' + str(bestNnAcc) + ' : ' + str(bestNnFeat))
