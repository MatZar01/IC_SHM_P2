# IC SHM 2021 Project 2 submission readme

This is a readme file describing the operation and processing of data for the purposes of implementing **Project 2** solution from [**The 2nd International Competition for Structural Health Monitoring**](https://sail.cive.uh.edu/ic-shm2021/). The work was done by me ([Mateusz Żarski, MSc](https://www.iitis.pl/en/node/3227)) and my associate ([Bartosz Wójcik, MSc](https://www.iitis.pl/en/person/bwojcik)) at the [Institute of Theoretical and Applied Informatics of the Polish Academy of Sciences](https://www.iitis.pl/en). 

##  Table of contents

* [General info](#general-info)
* [Dependencies](#dependencies)
* [Directory structure](#directory-structure)
* [Usage](#usage)
* [Use examples](#use-examples)

# General Info

For the purposes of the competition solution, we have created a robust pipeline of deep learning models using **Python 3** in which we utilized [Detectron2](https://github.com/facebookresearch/detectron2) framework for image semantic segmentation and fork of our own framework - [KrakN](https://github.com/MatZar01/KrakN) for training multiple image recognition models. Our pipeline of operations needed for performing all of the competition tasks, with addition of *Task 0* for masking background, presents itself as follows:

<img src="https://i.ibb.co/L5mQHVR/Fig7.png" alt="Our pipeline" width="650"/>

In total, we use four models of deep machine learning:

 - For masking the background - semantic segmentation,
 - For segmentation of construction elements,
 - For defect segmentation,
 - For damage state detection - image recognition.

A detailed description of the individual operations performed within the pipeline is described in the associated paper (will be uploaded after the competition ends). It also contains a description of the tests of other machine learning methods against which the solution from the project was compared.

## Dependencies

Our solution requires the following dependencies (packages in the latest version as of January 18, 2022, unless specified otherwise):

* TensorFlow == 1.12.0
* Detectron2
* Scikit-learn 
* Numpy == 1.16.2
* OpenCV == 4.4.0
* Matplotlib
* H5Py 
* Progressbar 
* Imutils 
* Pillow 
* PyGame 

Python version 3.8.10 was used, but different versions will also probably work fine (but we didn't check them).

Also please note, that strings containing paths to folders in our Python scripts may need to be changed in order to run properly on your system (we did all the work on Linux machine, so check your backslash).

## Directory structure

To use the solution we propose, a certain directory structure should be maintained, that is also consistent with out repository structure. The structure of the project with the names of the scripts is presented in the figure below:

<img src="https://i.ibb.co/MM9MdrM/Fig-repo.png" alt="Directory structure" width="600"/>

In the diagram, folders are marked in blue and Python scripts are marked in yellow. How to use each of them will be described in the next section.

## Usage

In order to use our solution with the dataset provided in Project or reproduce our results, cetrain steps have to be followed in order.

> 1. Split dataset to training and testing subsets.

First, dataset has to be split using `split_dataset.py` script. It will produce the split in 4:1 ratio and known pseudo-random algorithms' seed and place images in `dataset_reworked` directory. The script uses `.csv` files with image names provided in the Project.

> 2. Rework labels.

In the second steps labels have to be reworked with `rework_labels.py` so that they no longer are read by `cv2` library as RGB images but as 8-bit 1 channel images instead. This step is performed to make the images management a little bit easier, as they will be read as arrays from now on.

> 3. Make dataset for Task 0.

Now, `bcg_remover.py` have to be run in order to prepare the dataset to train background removal model. Images will be placed in `dataset_reworked_no_bcg` directory. The same images will later be used for Task 2.

> 4. Make dataset for Task 1 and Task 3

In the last step of dataset preparation, separate datasets for defect and damage state detection have to be prepared. In order to do so, run `comp_split.py` and `ds_dataset.py`. In the result, yet another set of training/testing images will be created in `./semantic_segmentation/dataset_comp` and `./damage_state_detection/dataset` directories. After this step, the creation of datasets is finally finished.

> 5. Train models for Task 0, 1 and 2.



## Use examples

Below is a video showing our solutions' pipeline in motion (it will redirect to external page).

<a href="https://streamable.com/qfx70h" title="Click me ;-)"><img src="https://i.ibb.co/RQgFSp2/front.png" alt="Click me ;-)" /></a>

Also, here are some images showing various tasks performed by our solution:

<img src="https://i.ibb.co/XWwqPY2/Fig1.jpg" alt="Task 0" width="800"/>

> Task 0: background masking

<img src="https://i.ibb.co/CM8gDgH/Fig4.jpg" alt="Task 1" width="500"/>

> Task 1: defect detection

<img src="https://i.ibb.co/HhgSvFP/Fig5.jpg" alt="Task 2" width="800"/>

> Task 2: element segmentation

<img src="https://i.ibb.co/8XXzj9k/Fig-repo-ds.png" alt="Task 3" width="800"/>

> Task 3: damage state detection

