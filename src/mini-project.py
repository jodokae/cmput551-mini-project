import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import utils

def paramSearch(X, y):

    nnParams = [{
        'activation':['logistic'],
        'hidden_layer_sizes':[(10,), (50,), (100,)],
        'alpha':[1e-4, 1e-3, 1e-2],
        'learning_rate_init':[1e-3, 1e-2, 1e-1],
        'beta_1':[0.1, 0.5, 0.9], 
        'beta_2':[0.1, 0.5, 0.9]
        }]
    
    treeParams = [{'criterion':['gini'],
        'splitter':['random'],
        'random_state':[None],
        'max_depth':[1, 10, 100], 
        'min_samples_split':[1e-5, 1e-4, 1e-3], 
        'min_samples_leaf':[1e-5, 1e-4, 1e-3],
        'min_weight_fraction_leaf':[1e-5, 1e-4, 1e-3],
        'max_features':['sqrt', 'log2'],
        'max_leaf_nodes':[100, 1000, 10000]
        }]
    
    print('Grid Search for Tree')
    clf = GridSearchCV(DecisionTreeClassifier(), treeParams, cv=5, verbose=1, n_jobs=-1)
    bestParamsTree = paramSearchCV(clf, X, y)
    
    print('Grid Search for Neural Net')
    clf = GridSearchCV(MLPClassifier(), nnParams, cv=5, verbose=1, n_jobs=-1)
    bestParamsNN = paramSearchCV(clf, X, y)
    
    return (bestParamsTree, bestParamsNN)
        
def paramSearchCV(clf, X, y):
    clf.fit(X, y)
    
    acc = clf.best_score_
    bestParams = clf.best_params_
    print(str(bestParams) + ': ' + str(acc))
    return bestParams


def compareAlgos(X, y, numruns, treeParams, nnParams):
    X_learn, X_test, y_learn, y_test = utils.splitData(X, y, numruns)
        
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
            print(algo)   
        
            algo.fit(X_learn[split], y_learn[split])

            y_pred = algo.predict(X_test[split])

            tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
        
            confusionMatrix[indexA][split][0] = tp
            confusionMatrix[indexA][split][1] = fp
            confusionMatrix[indexA][split][2] = fn
            confusionMatrix[indexA][split][3] = tn
            
    
    print('\nMean')
    utils.printMeanedMatrix(confusionMatrix[0])
    print('\nBayes')
    utils.printMeanedMatrix(confusionMatrix[1])
    print('\nTree')
    utils.printMeanedMatrix(confusionMatrix[2])
    print('\nNeural Network')
    utils.printMeanedMatrix(confusionMatrix[3])
    
    np.save('confusion.npy', confusionMatrix)
    print(confusionMatrix)
    

### MAIN ###
fullDataSet = True
print('Use complete Dataset: ' + str(fullDataSet))

print('Search for best parameters')

(X_train, X_test, y_train, y_test) = utils.loadAndSplit(fullDataSet)
(bestTreeFeat, bestNnFeat) = paramSearch(X_train, y_train)

print('\nTree Params: ' + str(bestTreeFeat))
print('NN Params: ' + str(bestNnFeat))

print('Run with best parameters')
compareAlgos(X_test, y_test, 10, bestTreeFeat, bestNnFeat)

print('\nTree Params: ' + str(bestTreeFeat))
print('NN Params: ' + str(bestNnFeat))

