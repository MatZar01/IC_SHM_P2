import sys

import cv2
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def ishow(image):
    plt.imshow(image)
    plt.show()

# select your task and label class - this is done only for defects, so manually typing eg. 'ds' is no no
tasks = ['crack', 'rebar', 'spall']
task = tasks[2]

mode = 'train'
mode = 'test'

IM_DIR = './dataset_reworked_no_bcg/{}'.format(mode)
LAB_DIR = './labels_reworked/{}/{}'.format(mode, task)
COMP_DIR = './labels_reworked/{}/component'.format(mode)
OUT_DIR = './dataset_comp/{}'.format(task)

# labels
# coponents
wall = 1
beam = 2
column = 3
window = 4 # exclude
balcon = 5 # exclude
slab = 6 # can be excluded, as it turned out that slabs are defect free
comp_list = [wall, beam, column, slab]
counter = 0

# now lets create our image-label pairs

im_pts = list(paths.list_images(IM_DIR))

# cycle images
for i in tqdm(range(len(im_pts))):
    pt = im_pts[i]
    image = cv2.imread(pt, -1)
    image_lab = cv2.imread('{}/{}'.format(LAB_DIR, pt.split('/')[-1]), -1)
    image_comp = cv2.imread('{}/{}'.format(COMP_DIR, pt.split('/')[-1]), -1)

    for component in comp_list:
        # select single type of component first
        single_comp_type = np.where(image_comp == component, 1, 0).astype('uint8')

        conts, hier = cv2.findContours(single_comp_type, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cont in conts:
            # select single component from selected type
            single_comp = np.zeros_like(single_comp_type)
            cv2.drawContours(single_comp, [cont], -1, 1, -1)
            x, y, w, h = cv2.boundingRect(cont)
            mask_im = np.zeros_like(single_comp_type)
            cv2.rectangle(mask_im, (x, y), (x + w, y + h), 1, -1)

            # get only bigger elements so we will not overflow CNN with garbage images
            if np.count_nonzero(mask_im) <= 244*244:
                continue

            # enlarge small elements
            if w < 200:
                x -= 50
                if x < 0:
                    x = 0
                w += 100
            if h < 200:
                y -= 50
                if y < 0:
                    y = 0
                h += 100

            # check for defects in frame and skip them if there are none
            if np.count_nonzero(cv2.bitwise_and(mask_im, image_lab)) == 0:
                #print('NO DEF')
                continue
            
            # save image-label pair
            cv2.imwrite('{}/{}/{}/{}.png'.format(OUT_DIR, mode, 'image', counter), image[y:y+h, x:x+w])
            cv2.imwrite('{}/{}/{}/{}.png'.format(OUT_DIR, mode, 'label', counter), image_lab[y:y+h, x:x+w])
            counter += 1
