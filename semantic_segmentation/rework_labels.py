# defects
import cv2
import numpy as np

# set your task - select train or test dataset and label class
# first we'll rework defect labels
DIR = 'test'
CLASS = 'spall'

LABEL_DIR = './dataset_init/label'
out_dir = './labels_reworked/{}/{}'.format(DIR, CLASS)

names_file = './dataset_reworked/{}_names.csv'.format(DIR)

names = np.genfromtxt(names_file, dtype=None, encoding=None)

# save new labels as 8-bit single channel images
for name in names:
    label_im = cv2.imread('{}/{}/{}'.format(LABEL_DIR, CLASS, name))
    #print(label_im)
    new_lab = np.where(label_im != 0, 1, 0)
    crc = np.count_nonzero(new_lab)
    '''if crc != 0:
        print(crc)
        cv2.imwrite('/home/mateusz/Pulpit/{}'.format(name), new_lab[:, :, 2])
    break'''
    cv2.imwrite('{}/{}'.format(out_dir, name), new_lab[:, :, 2])

print("Done")

#%%
# now components labels
import cv2
import numpy as np

DIR = 'train'
CLASS = 'component'

LABEL_DIR = './dataset_init/label'
out_dir = './labels_reworked/{}/{}'.format(DIR, CLASS)

names_file = './dataset_reworked/{}_names.csv'.format(DIR)

names = np.genfromtxt(names_file, dtype=None, encoding=None)

wall = 150
beam = 100
column = 186
window_frame = 133
window_pane = 206
balcony = 160
slab = 1

# again, save new labels
for name in names:
    label_im = cv2.imread('{}/{}/{}'.format(LABEL_DIR, CLASS, name))[:, :, 0]

    wall_lab = np.where(label_im == wall, 1, 0)
    beam_lab = np.where(label_im == beam, 2, 0)
    column_lab = np.where(label_im == column, 3, 0)
    win_l_1 = np.where(label_im == window_frame, 4, 0)
    win_l_2 = np.where(label_im == window_pane, 5, 0)
    balcon_lab = np.where(label_im == balcony, 6, 0)
    slab_lab = np.where(label_im == slab, 7, 0)

    new_lab = wall_lab + beam_lab + column_lab + win_l_1 + win_l_2 + balcon_lab + slab_lab
    cv2.imwrite('{}/{}'.format(out_dir, name), new_lab)

print("Done")
#%%
# at last - damage state labels
import cv2
import numpy as np
import matplotlib.pyplot as plt

DIR = 'test'
CLASS = 'ds'

LABEL_DIR = './label'
out_dir = './labels_reworked/{}/{}'.format(DIR, CLASS)

names_file = './dataset_reworked/{}_names.csv'.format(DIR)

names = np.genfromtxt(names_file, dtype=None, encoding=None)

no_dmg = (0, 255, 0)[::-1]
light_dmg = (150,250,0)[::-1]
mod_dmg = (255,225,50)[::-1]
sev_dmg = (255,0,0)[::-1]

# that one is tricky because of the incredibly bad way of delivering the data
for name in names:
    label_im = cv2.imread('{}/{}/{}'.format(LABEL_DIR, CLASS, name), -1)

    no_lab = np.where((label_im[:, :, 0] == no_dmg[0]) & (label_im[:, :, 1] == no_dmg[1]) == True, 1, 0)
    light_lab = np.where((label_im[:, :, 0] == light_dmg[0]) & (label_im[:, :, 1] == light_dmg[1]) == True, 2, 0)
    mod_lab = np.where((label_im[:, :, 0] == mod_dmg[0]) & (label_im[:, :, 1] == mod_dmg[1]) == True, 3, 0)
    sev_lab = np.where((label_im[:, :, 0] == sev_dmg[0]) & (label_im[:, :, 1] == sev_dmg[1]) == True, 4, 0)

    new_lab = no_lab + light_lab + mod_lab + sev_lab
    cv2.imwrite('{}/{}'.format(out_dir, name), new_lab)

print("Done")
