import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable 

def learnAndPredict(learner, X_learn, y_learn, X_test, y_test):
    learner.fit(X_learn, y_learn)

    y_pred = learner.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return (tn, fp, fn, tp)
    

def printMeanedMatrix(matrix):
    means = np.mean(matrix, axis=0)
    printMatrix(means[0], means[1], means[2], means[3])

def printMatrix(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    correctLabeled = tp + tn
    acc = correctLabeled / total
    
    print("Number of mislabeled points out of a total %d points : %d" % (total,(total-correctLabeled)))
    
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, total])
    print(x)
    
    print('Acc: ' + str(acc))
    
    
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
    
def paramSearch(X, y):

    nnParams = [{
        'activation':['logistic'],
        #'hidden_layer_sizes':[(10,), (50,), (100,)],
        #'alpha':[1e-4, 1e-3, 1e-2],
        #'learning_rate_init':[1e-3, 1e-2, 1e-1],
        #'beta_1':[0.1, 0.5, 0.9], 
        #'beta_2':[0.1, 0.5, 0.9]
        }]
    
    treeParams = [{'criterion':['gini'],
            'splitter':['random'],
            'random_state':[None],
            #'max_depth':[1, 10, 100], 
            #'min_samples_split':[1e-5, 1e-4, 1e-3], 
            #'min_samples_leaf':[1e-5, 1e-4, 1e-3],
            #'min_weight_fraction_leaf':[1e-5, 1e-4, 1e-3],
            'max_features':['sqrt', 'log2'],
            #'max_leaf_nodes':[100, 1000, 10000]
        }]
    
    print('Grid Search for Tree')
    clf = GridSearchCV(DecisionTreeClassifier(), treeParams, cv=5, verbose=1, n_jobs=-1)
    
    clf.fit(X, y)
    
    acc = clf.best_score_
    bestParamsTree = clf.best_params_
    print(str(bestParamsTree) + ': ' + str(acc))
    
    print('Grid Search for Neural Net')
    clf = GridSearchCV(MLPClassifier(), nnParams, cv=5, verbose=1, n_jobs=-1)
    clf.fit(X, y)
    
    acc = clf.best_score_
    bestParamsNN = clf.best_params_
    print(str(bestParamsNN) + ': ' + str(acc))
    
    return (bestParamsTree, bestParamsNN)
    
def loadAndSplit(fullSet):

    X, y = load(fullSet)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
    return (X_train, X_test, y_train, y_test)
    
def splitData(X, y):
    print ('Splitting Data')

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1)

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
    
def compareAlgos(X, y, numruns, treeParams, nnParams):
    X_learn, X_test, y_learn, y_test = splitData(X, y)
    
    numruns = 5
    
    algorithms = [
        DummyClassifier(strategy='most_frequent'),
        GaussianNB(),
        DecisionTreeClassifier(**treeParams),
        MLPClassifier(**nnParams)
    ]
    
    numalgos = len(algorithms)
    
    confusionMatrix = np.zeros((numalgos, numruns, 4))
        
    for split in range(numruns):
        print('Run ' + str(split+1) + '/' + str(numruns))   
        
        for indexA, algo in enumerate(algorithms):    
        
            tn, fp, fn, tp = learnAndPredict(algo, X_learn[split], y_learn[split], X_test[split], y_test[split])
        
            confusionMatrix[indexA][split][0] = tp
            confusionMatrix[indexA][split][1] = fp
            confusionMatrix[indexA][split][2] = fn
            confusionMatrix[indexA][split][3] = tn
        
    
    print('\nMean')
    printMeanedMatrix(confusionMatrix[0])
    print('\nBayes')
    printMeanedMatrix(confusionMatrix[1])
    print('\nTree')
    printMeanedMatrix(confusionMatrix[2])
    print('\nNeural Network')
    printMeanedMatrix(confusionMatrix[3])
    


print('Search for best parameters')
(X_train, X_test, y_train, y_test) = loadAndSplit(False)
(bestTreeFeat, bestNnFeat) = paramSearch(X_train, y_train)
print('Run with best parameters')
compareAlgos(X_test, y_test, 5, bestTreeFeat, bestNnFeat)

print('\nTree Params: ' + str(bestTreeFeat))
print('NN Params: ' + str(bestNnFeat))


