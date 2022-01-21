import numpy as np
train_names = np.genfromtxt('./train.csv', dtype=None, encoding=None)
train_names = np.delete(train_names, 57)

dummy_y = np.asarray(range(train_names.size))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_names, dummy_y, test_size=0.2, random_state=314)
#%%
# save split to new .csv
import csv

with open("./dataset_reworked/train_names.csv", 'w') as f:
    for i in range(X_train.size):
        csv.writer(f).writerow([X_train[i]])

with open("./dataset_reworked/test_names.csv", 'w') as f:
    for i in range(X_test.size):
        csv.writer(f).writerow([X_test[i]])

#%%
# and copy images to dataset_reworked dir
from shutil import copyfile

for name in X_train:
    copyfile(f'./dataset_init/image/{name}', f'./dataset_reworked/train/{name}')

for name in X_test:
    copyfile(f'./dataset_init/image/{name}', f'./dataset_reworked/test/{name}')
