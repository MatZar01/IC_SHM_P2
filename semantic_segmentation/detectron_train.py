"""In this file training and inference with trained model 
can be performed. First task is picked, then training parameters,
and training itself is performed. 
After training you can run inference and visualization of data segmentation"""

# make initial detectron imports
import detectron2
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries for arrays, images and jsons
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
# you can train model with and without background masking
background_mask = False
#%%

# register dataset
from detectron2.data.datasets import load_sem_seg

def get_train():
    if task == 'bg_fg' or not background_mask:
        train = load_sem_seg('./labels_reworked/train/{}'.format(task), './dataset_reworked/train', gt_ext='png',
                            image_ext='png')
    else:
        train = load_sem_seg('./labels_reworked/train/{}'.format(task), './dataset_reworked_no_bcg/train', gt_ext='png',
                            image_ext='png')

    return train

classes = task_dict[task]

DatasetCatalog.register('{}_train'.format(task), lambda: get_train())
MetadataCatalog.get('{}_train'.format(task)).set(stuff_classes=classes)
#%%

# apply model hyperparameters
cfg = get_cfg()
# fine-tune mask rcnn trained initially on ImageNet
from detectron2.engine import DefaultTrainer

cfg.merge_from_file(model_zoo.get_config_file("Misc/semantic_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ('{}_train'.format(task),)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 15 * 1520   # pick iterations
cfg.SOLVER.STEPS = []        # do not decay learning rate this time
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(classes)
cfg.OUTPUT_DIR = './output_{}_seg'.format(task) # in this path trained model and checkpoints will be saved
#%%

# train model - uncomment cell only when training, or you will overwrite your model!
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
#%%

# or use pretrained model for inference
cfg.MODEL.WEIGHTS = os.path.join("./output_{}_seg/model_final.pth".format(task))  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
#%%

# visualize some random images
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from imutils import paths

ds_type = 'test'
im_pts = list(paths.list_images('/home/mateusz/Pulpit/IC_SHM_21/dataset_reworked/{}/'.format(ds_type)))

for i in range(10):
    pt = random.choice(im_pts)
    im = cv2.imread(pt)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.5,
                   instance_mode=ColorMode.SEGMENTATION,
                   )

    x = outputs["sem_seg"].argmax(dim=0)
    out = v.draw_sem_seg(x.to("cpu"))
    # save predictions as images
    cv2.imwrite('/home/mateusz/Pulpit/sender/{}.png'.format(i), im)
    cv2.imwrite('/home/mateusz/Pulpit/sender/{}_pred.png'.format(i), cv2.resize(out.get_image(), im.shape[:-1][::-1]))
#%%

# make labels for later tasks - simply run inference
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from imutils import paths

ds_type = 'train'

im_pts = list(paths.list_images('/home/mateusz/Pulpit/IC_SHM_21/dataset_comp/{}/{}/image'.format(task, ds_type)))

for pt in im_pts:
    im = cv2.imread(pt)
    outputs = predictor(im)
    outputs = outputs['sem_seg'].argmax(dim=0).to('cpu').numpy()
    # and save last activation map
    cv2.imwrite(pt.replace('/dataset_reworked_no_bcg/{}/'.format(ds_type), '/result_dmg_masks/{}/{}/'.format(ds_type, task)), outputs)
