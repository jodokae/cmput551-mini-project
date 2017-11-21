import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
<<<<<<< HEAD
from prettytable import PrettyTable

print ('Loading Data')
=======
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742

A = np.load('A.npy')
y = np.load('y.npy')

<<<<<<< HEAD
print ('Encoding Data')

=======
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(A)

A_missingEncoded = imp.transform(A)
#scale(A_missingEncoded)

<<<<<<< HEAD
print ('Splitting Data')

=======
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
(A_learn, A_test) = np.split(A_missingEncoded, 2)
(y_learn, y_test) = np.split(y, 2)


#print(A_missingEncoded)
#print(y)

print ('Comp: 0 / 1: ' + str((y == 0).sum() / (y == 1).sum()))
print ('Lear: 0 / 1: ' + str((y_learn == 0).sum() / (y_learn == 1).sum()))
print ('Test: 0 / 1: ' + str((y_test == 0).sum() / (y_test == 1).sum()))

gnb = GaussianNB()
gnb.fit(A_learn, y_learn)

print(gnb.get_params())

y_pred = gnb.predict(A_test)
print("Number of mislabeled points out of a total %d points : %d" % (A_test.shape[0],(y_test != y_pred).sum()))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
<<<<<<< HEAD
x = PrettyTable()
x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
x.add_row(["True (Pred)",tp, fp, tp+fp])
x.add_row(["False (Pred)",fn, tn, fn + tn])
x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
print(x)
=======
print(tp)
print(fp)
print(fn)
print(tn)
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742

from sklearn import tree


print('\n Decision Tree \n\n')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(A_learn, y_learn)

y_pred = clf.predict(A_test)

print("Number of mislabeled points out of a total %d points : %d" % (A_test.shape[0],(y_test != y_pred).sum()))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
<<<<<<< HEAD
x = PrettyTable()
x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
x.add_row(["True (Pred)",tp, fp, tp+fp])
x.add_row(["False (Pred)",fn, tn, fn + tn])
x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
print(x)
=======
print(tp)
print(fp)
print(fn)
print(tn)
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742

print('\n Neural Net \n\n')


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.fit(A_learn, y_learn)
y_pred = clf.predict(A_test)

print("Number of mislabeled points out of a total %d points : %d" % (A_test.shape[0],(y_test != y_pred).sum()))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
<<<<<<< HEAD
x = PrettyTable()
x.field_names = ["", "True (Real)", "False (Real)", "Sum"]
x.add_row(["True (Pred)",tp, fp, tp+fp])
x.add_row(["False (Pred)",fn, tn, fn + tn])
x.add_row(["Sum", tp+fn, fp+tn, tp+fn+tn+fp])
print(x)
=======
print(tp)
print(fp)
print(fn)
print(tn)
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
