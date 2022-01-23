"""
You can test your trained classifier 
with this script
"""

import platform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle
import h5py
import os
import sys

# set model path
model_path = '/home/mateusz/Pulpit/KrakN/network_trainer/resnet_clf.cpickle'

# open database
testPath = '/home/mateusz/Pulpit/KrakN/network_trainer/database/resnet_test_s_1.hdf5'
test_db = h5py.File(testPath, "r")

X_test = test_db["features"]
y_test = test_db["labels"]

# open model
classifier = pickle.load(open(model_path, 'rb'))

# make predictions
preds = classifier.predict(X_test)
avg_score = classifier.score(X_test, y_test) * 100
conf_matrix = confusion_matrix(y_test, preds)
class_acc = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
class_acc = class_acc * 100
f1 = f1_score(y_test, preds, average='weighted') * 100

# and print scores
print('Model: {}'.format(model_path.split('/')[-1].split('_')[0]))
print(f'AVG accuracy: {avg_score}')
print(f'Accuracy per class: {class_acc}')
print(f'F1 score: {f1}')
