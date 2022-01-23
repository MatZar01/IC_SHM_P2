"""
This file is for classifier training after
feature extraction
"""

#!python3
try:
    import platform
    from sklearn.linear_model import LogisticRegression # we'll use single layer perceptron
    from sklearn.model_selection import GridSearchCV # with various hyperparameters
    from sklearn.metrics import classification_report
    import pickle
    import h5py
    import os
    import sys
except ImportError as e:
    print(e)
    print("One or more dependencies missing!\nOpen README file to check required dependencies.")
    if platform.system() == 'Linux':
        print("\nYou can install all dependencies using install_dependencies.sh")
    else:
        print("\nYou can install all dependencies using install_dependencies.bat")
    sys.exit()

jobs = 1

# open database to resnet features
trainPath = './database/resnet_train_s_1.hdf5'
testPath = './database/resnet_test_s_1.hdf5'
train_db = h5py.File(trainPath, "r")
test_db = h5py.File(testPath, "r")

# and set the training / testing split index - you can change 1.0 to smaller value if features 
# do not fit in RAM
i = int(1.0*len(train_db["features"]))
X_train = train_db["features"][:i]
y_train = train_db["labels"][:i]
X_test = test_db["features"][:i]
y_test = test_db["labels"][:i]
#%%

modelPath = '{}_clf.cpickle'.format(trainPath.split('/')[-1].split('_')[0])
if os.path.exists(modelPath):
    os.remove(modelPath)

# train Logistic Regression classifier
print("Tuning hyperparameters...")
params = {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
model = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=3, n_jobs=jobs, verbose=20)
model.fit(X_train, y_train)
print("Best hyperparameter: {}".format(model.best_params_))

# save model to disk
print("Saving model...")
f = open(modelPath, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
#%%
# evaluate model
print("Evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))

train_db.close()
test_db.close()
