import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score   
import itertools
    

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def learnAndPredict(learner, X_learn, y_learn, X_test, y_test, printResults=False):
    learner.fit(X_learn, y_learn)

    y_pred = learner.predict(X_test)
    
    if printResults == True:
        printRes(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return (accuracy_score(y_test, y_pred), tn, fp, fn, tp)
    

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
    

def run(fullData, printResults, numruns, paramSearch=True, bestTreeParams={}, bestNeuralParams={}):

    X, y = load(fullData)

    print ('Splitting Data')

    sss = StratifiedShuffleSplit(n_splits=numruns, test_size=0.1)

    X_learn = []
    X_test = []
    y_learn = []
    y_test = []

    for train_index, test_index in sss.split(X, y):
        X_learn.append(X[train_index])
        X_test.append(X[test_index])
        y_learn.append(y[train_index])
        y_test.append(y[test_index])


    if paramSearch == True:
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
    else:
        treeParams = [bestTreeParams]
        nnParams = [bestNeuralParams]
        
    numTreeParams = len(treeParams)
    numNnParams = len(nnParams)
    numMeanParams = 1
    numBayesParams = 1
    
    meanAcc = np.zeros((numruns, numMeanParams))
    bayesAcc = np.zeros((numruns, numBayesParams))
    treeAcc = np.zeros((numruns, numTreeParams))
    nnAcc = np.zeros((numruns, numNnParams))
    
    meanMatrix = np.zeros((numruns, 4))
    bayesMatrix = np.zeros((numruns, 4))
    treeMatrix = np.zeros((numruns, 4))
    nnMatrix = np.zeros((numruns, 4))

    print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))

    for split in range(numruns):

        print('Run ' + str(split+1) + '/' + str(numruns))
        
        
        print('Mean')

        pred = 0 if ((y_learn[split] == 0).sum()) > ((y_learn[split] ==1).sum()) else 1
        y_pred = np.zeros(y_test[split].shape) + pred
        
        meanAcc[split][0] = accuracy_score(y_test[split], y_pred) 
        tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
        
        print(str(1) + '/' + str(1) + ': ' + str(meanAcc[split][0]))
        
        if printResults == True:
            meanMatrix[split][0] = tp
            meanMatrix[split][1] = fp
            meanMatrix[split][2] = fn
            meanMatrix[split][3] = tn
        
        
                    
        print('Gaussian Naive Bayes')

        gnb = GaussianNB()
        bayesAcc[split][0], tn, fp, fn, tp = learnAndPredict(gnb, X_learn[split], y_learn[split], X_test[split], y_test[split])

        print(str(1) + '/' + str(1) + ': ' + str(bayesAcc[split][0]))
          
        if printResults == True:
            bayesMatrix[split][0] = tp
            bayesMatrix[split][1] = fp
            bayesMatrix[split][2] = fn
            bayesMatrix[split][3] = tn
        
        print('Decision Tree Gini')   
        
        for index, params in enumerate(treeParams):
            clf = DecisionTreeClassifier(**params)
        
            #print(clf.get_params())
       
            acc, tn, fp, fn, tp = learnAndPredict(clf, X_learn[split], y_learn[split], X_test[split], y_test[split])
            
            print(str(index+1) + '/' + str(numTreeParams) + ': ' + str(acc))
            
            treeAcc[split][index] = acc
            
            if printResults == True:
                treeMatrix[split][0] = tp
                treeMatrix[split][1] = fp
                treeMatrix[split][2] = fn
                treeMatrix[split][3] = tn
                       
        
        print('Neural Net')
        
        for index, params in enumerate(nnParams):
            clf = MLPClassifier(**params)
            
            acc, tn, fp, fn, tp = learnAndPredict(clf, X_learn[split], y_learn[split], X_test[split], y_test[split])
            
            print(str(index+1) + '/' + str(numNnParams) + ': ' + str(acc))
            
            nnAcc[split][index] = acc
        
            if printResults == True:
                nnMatrix[split][0] = tp
                nnMatrix[split][1] = fp
                nnMatrix[split][2] = fn
                nnMatrix[split][3] = tn

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

    if printResults==False:
        print('Mean: Acc ' + str(bestMeanAcc))
        print('Bayes: Acc ' + str(bestBayesAcc))
        print('Tree: Acc ' + str(bestTreeAcc) + ' : ' + str(bestTreeFeat))
        print('NN: Acc ' + str(bestNnAcc) + ' : ' + str(bestNnFeat))
    else:
        print('\nMean')
        printMeanedMatrix(meanMatrix)
        print('\nBayes')
        printMeanedMatrix(bayesMatrix)
        print('\nTree')
        printMeanedMatrix(treeMatrix)
        print('\nNeural Network')
        printMeanedMatrix(nnMatrix)

    return(bestTreeFeat, bestNnFeat)


(bestTreeFeat, bestNnFeat) = run(False, False, 1)
run(False, True, 2, False, bestTreeFeat, bestNnFeat)

print('\nTree Params: ' + str(bestTreeFeat))
print('NN Params: ' + str(bestNnFeat))


