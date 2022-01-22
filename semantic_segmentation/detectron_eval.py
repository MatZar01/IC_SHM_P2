"""
This file starts off similarly to detectron_train.py.
However it will not train new model and only run inference with existing ones.
And save its metrics.
"""

# make inital detectron imports
import detectron2
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
#%%

# setup task dict
task_dict = {
  'bg_fg': ['background', 'foreground'],
  'crack': ['background', 'crack'],
  'spall': ['background', 'spall'],
  'rebar': ['background', 'rebar'],
  'component': ["background", "wall", "beam", "column", "window frame", "window pane", "balcony", "slab"],
  'ds': ['background', 'none', 'light', 'moderate', 'severe']
}

# setup task
task = 'component'
# again you can run tests on images with unmasked background
background_mask = False
#%%

# this wime use pretrained model only
cfg.MODEL.WEIGHTS = os.path.join("./output_{}_seg/model_final.pth".format(task))  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
#%%

# and initialize evaluator
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper

seg_evaluator = SemSegEvaluator(cfg.DATASETS.TRAIN[0],distributed=True,output_dir=cfg.OUTPUT_DIR,ignore_label=[])
eval_loader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0],mapper = DatasetMapper(cfg, is_train=False, augmentations=[]))
results = inference_on_dataset(predictor.model, eval_loader, seg_evaluator)
#%%

# initialize JSON to dump the results
import json

if background_mask:
  res_file_path = './results/results_{}.json'.format(task)
else:
  res_file_path = './results/results_{}_no_bcg_mask.json'.format(task)

results['{}_eval'.format(task)] = results.pop('sem_seg')
with open(res_file_path, 'w') as out_file:
  json.dump(results, out_file, indent=4)
#%%

# here we also checked how many images did not have any rebar labels
# and how small labeled areas were in others
# spoiler: many and very small
import cv2
from imutils import paths
import numpy as np

pts = list(paths.list_images('./labels_reworked/train/rebar'))
w_rebar = 0
wo_rebar = 0
min_rebar = 100000
max_rebar = 0

for pt in pts:
  im = cv2.imread(pt, -1)
  ct = np.count_nonzero(im)

  if ct != 0:
    w_rebar += 1
    if ct > max_rebar:
      max_rebar = ct
    if ct < min_rebar:
      min_rebar = ct
  else:
    wo_rebar += 1

print(f"w_rebar: {w_rebar}\nwo_rebar: {wo_rebar}")
print(f'min rebar: {min_rebar} px - {min_rebar/(im.size/3)*100}%\nmax rebar: {max_rebar} px - {max_rebar/(im.size/3)*100}%')
