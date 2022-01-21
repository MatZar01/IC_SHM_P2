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

# select your task
mode = 'test'

COMP_IM_DIR = './labels_reworked/{}/component'.format(mode)
IM_DIR = './dataset_reworked_no_bcg/{}'.format(mode)
DS_DIR = '../damage_state_detection/dataset/{}/'.format(mode)

# list coponents
wall = 1
beam = 2
column = 3
window = 4 # ignore
balcon = 5 # ignore
slab = 6
comp_list = [wall, beam, column, slab]
# damages
no_dmg = 0
light_dmg = 1
mod_dmg = 2
sev_dmg = 3

counter = 0

im_pts = list(paths.list_images(COMP_IM_DIR))

im_pts = [im_pts[5]]

# cycle images
for i in tqdm(range(len(im_pts))):
    pt = im_pts[i]
    image = cv2.imread(pt, -1)
    image_ds = cv2.imread('{}/{}'.format(DS_DIR, pt.split('/')[-1]), -1)
    image_src = cv2.imread('{}/{}'.format(IM_DIR, pt.split('/')[-1]), -1)
    image_tmp = image_src.copy()

    for component in comp_list:
        # select single component type
        comp_img = np.where(image==component, 1, 0).astype('uint8')

        conts, hier = cv2.findContours(comp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cont in conts:
            # select single component of single type
            temp_im = np.zeros_like(image)
            cv2.drawContours(temp_im, [cont], -1, 1, -1)

            if np.count_nonzero(temp_im) <= 64*64:
                continue

            # take DS and element type reading
            ds_pt = random.choice(np.argwhere(temp_im == 1))
            element_ds = image_ds[ds_pt[0]][ds_pt[1]].astype('uint8')
            element_type = component
            if element_ds == 0 or element_ds == 5:
                print(f"\nDS error:\nComponent: {component}, DS: {element_ds}, File:{pt.split('/')[-1]}\n\n")
                continue

            # get minimum size rectangle with component
            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            box = np.where(box < 0, 0, box).astype('float32')

            # this time warp area to 224x224 rectangle
            dst_pts = np.array([[0, 0], [224, 0], [224, 224], [0, 224]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(box, dst_pts)
            warp = cv2.warpPerspective(image_src, M, (224, 224))
            
            # and save images to output folder
            counter += 1
            cv2.imwrite('{}/{}/{}/{}.png'.format(DS_DIR, mode, element_ds, counter), warp)
