# here we will remove the background for fg/bcg segmentation

import cv2
import numpy as np
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt

# set your dataset correctly to train or test
dir = 'train'

image_dir = './dataset_reworked/{}'.format(dir)

im_paths = list(paths.list_images(image_dir))

# rework each image
for i in tqdm(range(len(im_paths))):
	pt = im_paths[i]

	image = cv2.imread(pt)
	component = cv2.imread(pt.replace(image_dir, './label/component'))
	component_any = np.where(component != (70, 70, 70), 255, 0)
	background = np.where(component_any != 255, 255, 0)
  
  # and save it in labels_reworked dir
	cv2.imwrite(pt.replace(image_dir, './labels_reworked/background'), background)
	cv2.imwrite(pt.replace(image_dir, './labels_reworked/foreground'), component_any)

print('Done')
