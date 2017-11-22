import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score

print ('Loading Data')

A = np.load('A_small.npy')
y = np.load('y_small.npy')

print ('Encoding Data')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(A)

X = imp.transform(A)

print ('Splitting Data')

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=0)

X_learn = []
X_test = []
y_learn = []
y_test = []

for train_index, test_index in sss.split(X, y):
    X_learn.append(X[train_index])
    X_test.append(X[test_index])
    y_learn.append(y[train_index])
    y_test.append(y[test_index])


print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))

for split in range(len(X_learn)):
    
    print('\n Mean \n\n')

    pred = 0 if ((y_learn[split] == 0).sum()) > ((y_learn[split] ==1).sum()) else 1
    
    y_pred = np.zeros(y_test[split].shape) + pred
    
    print("Number of mislabeled points out of a total %d points : %d" % (X_test[split].shape[0],(y_test[split] != y_pred).sum()))
    tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
    print(x)
    
    print('Acc: ' + str(accuracy_score(y_test[split], y_pred)))
    
        
    print('\n Gaussian Naive Bayes \n\n')

    gnb = GaussianNB()
    gnb.fit(X_learn[split], y_learn[split])

    print(gnb.get_params())

    y_pred = gnb.predict(X_test[split])
    print("Number of mislabeled points out of a total %d points : %d" % (X_test[split].shape[0],(y_test[split] != y_pred).sum()))

    tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
    print(x)
    
    print('Acc: ' + str(accuracy_score(y_test[split], y_pred)))

    print('\n Decision Tree \n\n')

    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, 
            min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    
    clf = clf.fit(X_learn[split], y_learn[split])

    y_pred = clf.predict(X_test[split])

    print("Number of mislabeled points out of a total %d points : %d" % (X_test[split].shape[0],(y_test[split] != y_pred).sum()))
    tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
    print(x)
        
    print('Acc: ' + str(accuracy_score(y_test[split], y_pred)))

    print('\n Neural Net \n\n')

    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)

    clf.fit(X_learn[split], y_learn[split])
    y_pred = clf.predict(X_test[split])

    print("Number of mislabeled points out of a total %d points : %d" % (X_test[split].shape[0],(y_test[split] != y_pred).sum()))
    tn, fp, fn, tp = confusion_matrix(y_test[split], y_pred).ravel()
    x = PrettyTable()
    x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
    x.add_row(["True (Pred)",tp, fp, tp+fp])
    x.add_row(["False (Pred)",fn, tn, fn + tn])
    x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
    print(x)
    
    print('Acc: ' + str(accuracy_score(y_test[split], y_pred)))
