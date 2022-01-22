"""
In this file a few of shallow ML alhorithms are trained 
on data extracted as 1D feature vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# import data from csvs
train_data = np.loadtxt('./NN_dataset_train.csv', delimiter=',', skiprows=1)
test_data = np.loadtxt('./NN_dataset_test.csv', delimiter=',', skiprows=1)

X_train = train_data[:, 0:-1]
y_train = train_data[:, -1]

X_test = test_data[:, 0:-1]
y_test = test_data[:, -1]

print(f'Data shapes:\nTrain data: {X_train.shape}, labels: {y_train.shape}\n\
Test data: {X_test.shape}, labels: {y_test.shape}')

# train decision tree
classifier = tree.DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# check tree score
score_train = classifier.score(X_train, y_train) * 100
score_test = classifier.score(X_test, y_test) * 100
preds_tree = classifier.predict(X_test)
matrix_tree = confusion_matrix(y_test, preds_tree)
tree_acc_class = matrix_tree.diagonal()/matrix_tree.sum(axis=1) * 100
f1_tree = f1_score(y_test, preds_tree, average='weighted') * 100

print(f'Scores obtained:\nTrain set: {score_train}\nTest set: {score_test}')
print(f'Tree classes: {classifier.classes_} with depth: {classifier.get_depth()}')
print('Accuracy per class: {}'.format(tree_acc_class))
print('Average F1 score: {}'.format(f1_tree))


# train random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=200, max_leaf_nodes=25, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)

# score random forest
forest_score_train = forest.score(X_train, y_train) * 100
forest_score_test = forest.score(X_test, y_test) * 100
preds_forest = forest.predict(X_test)
matrix_forest = confusion_matrix(y_test, preds_forest)
forest_acc_class = matrix_forest.diagonal()/matrix_forest.sum(axis=1) * 100
f1_forest = f1_score(y_test, preds_forest, average='weighted') * 100

print(f'Scores obtained:\nTrain set: {forest_score_train}\nTest set: {forest_score_test}')
print(f'Forest classes: {forest.classes_} with {len(forest.estimators_)} estimators')
print(f'Accuracy per class: {forest_acc_class}')
print('Average F1 score: {}'.format(f1_forest))


# train Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_model = GaussianNB()
model_prediction = naive_model.fit(X_train, y_train)

nb_score_train = naive_model.score(X_train, y_train) * 100
nb_score_test = naive_model.score(X_test, y_test) * 100
preds_nb = naive_model.predict(X_test)
matrix_nb = confusion_matrix(y_test, preds_nb)
nb_acc_class = matrix_nb.diagonal()/matrix_nb.sum(axis=1) * 100
f1_nb = f1_score(y_test, preds_nb, average='weighted') * 100

print(f'Scores obtained:\nTrain set: {nb_score_train}\nTest set: {nb_score_test}')
print(f'Accuracy per class: {nb_acc_class}')
print('Average F1 score: {}'.format(f1_nb))


# try to normalize Data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

naive_model_norm = GaussianNB()
model_prediction_norm = naive_model_norm.fit(X_train_norm, y_train)

nb_score_train_norm = naive_model_norm.score(X_train_norm, y_train) * 100
nb_score_test_norm = naive_model_norm.score(X_test_norm, y_test) * 100
preds_nbm = naive_model_norm.predict(X_test)
matrix_nbm = confusion_matrix(y_test, preds_nbm)
nbm_acc_class = matrix_nbm.diagonal()/matrix_nbm.sum(axis=1) * 100
f1_nbn = f1_score(y_test, preds_nbm, average='weighted') * 100

print(f'Scores obtained:\nTrain set: {nb_score_train_norm}\nTest set: {nb_score_test_norm}')
print(f'Accuracy per class: {nbm_acc_class}')
print('Average F1 score: {}'.format(f1_nbn))
