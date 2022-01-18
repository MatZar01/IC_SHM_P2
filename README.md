# IC SHM 2021 Project 2 submission readme

This is a readme file describing the operation and processing of data for the purposes of implementing **Project 2** solution from [**The 2nd International Competition for Structural Health Monitoring**](https://sail.cive.uh.edu/ic-shm2021/). The work was done by me ([Mateusz Żarski, MSc](https://www.iitis.pl/en/node/3227)) and my associate ([Bartosz Wójcik, MSc](https://www.iitis.pl/en/person/bwojcik)) at the [Institute of Theoretical and Applied Informatics of the Polish Academy of Sciences](https://www.iitis.pl/en). 

##  Table of contents

* [General info](#general-info)
* [Dependencies](#dependencies)
* [Directory structure](#directory-structure)
* [Usage](#usage)
* [Use examples](#use-examples)
* [Acknowledgments](#acknowledgments)

# General Info

For the purposes of the competition solution, we have created a robust pipeline of deep learning models using **Python 3** in which we utilized [Detectron2](https://github.com/facebookresearch/detectron2) framework for image semantic segmentation and fork of our own framework -- [KrakN](https://github.com/MatZar01/KrakN) for training multiple image recognition models. Our pipeline of operations needed for performing all of the competition tasks presents itself as follows:

![Our pipeline](https://i.ibb.co/L5mQHVR/Fig7.png)

In total, we use four models of deep machine learning:

 - For masking the background -- semantic segmentation,
 - For segmentation of construction elements,
 - For defect segmentation,
 - For damage state detection -- image recognition.

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

## Directory structure

To use the solution we propose, a certain directory structure should be maintained, that is also consistent with out repository structure. The structure of the project with the names of the scripts is presented in the figure below:

![Directory structure](https://i.ibb.co/MM9MdrM/Fig-repo.png)



> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).


