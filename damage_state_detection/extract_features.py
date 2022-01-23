#!python3

"""
This script allows for making .hdf5 database
for classifier training. It uses ResNet50 as feature extractor
but you can change it, if you know what you're doing.
"""

try:
    import platform
    import os
    
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    import sys
    sys.path.append('.{}utilities{}io'.format(os.path.sep, os.path.sep))

    from hdf5_dataset_writer import HDF5DatasetWriter
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2 # if you are changing this
    from tensorflow.keras.applications.resnet_v2 import preprocess_input  # change this as well
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.preprocessing.image import load_img
    from sklearn.preprocessing import LabelEncoder
    from imutils import paths
    import numpy as np
    import progressbar
    import random
    import sys
except ImportError as e:
    print(e)
    print("One or more dependencies missing!\nOpen README file to check required dependencies.")
    if platform.system() == 'Linux':
        print("\nYou can install all dependencies using 'sudo chmod +x ./install_dependencies.sh & ./install_"
              "dependencies.sh' command in KrakN directory.")
    else:
        print("\nYou can install all dependencies using install_dependencies.bat in KrakN directory")
    sys.exit()

# set/check dataset & output path, delete previous output if exists
while True:
    datasetPath = input("Enter path to database directory (no quotes):\n")
    if not os.path.exists(datasetPath):
        print("Dataset at {}\nDoes not exist.".format(datasetPath))
    else:
        break

outputPath = r".{}database".format(os.path.sep)
if not os.path.exists(outputPath):
    os.mkdir('.{}database'.format(os.path.sep))
batchSize = 2
bufferSize = 1000

# load images and shuffle them
print("Loading images...")
imagePaths = list(paths.list_images(datasetPath))
random.shuffle(imagePaths)
print("{} images loaded".format(len(imagePaths)))

# set scale factor as 1, we don't need it here
scale = 1

# extract class labels and encode them
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load ResNet50V2 network excluding final FC layers
print("loading network...")
model = ResNet50V2(weights="imagenet", include_top=False)

# initialize dataset writer with correct features size - if you changed Resnet to something else
dataset = HDF5DatasetWriter((len(imagePaths), 2048 * 7 * 7), outputPath + os.path.sep +"resnet_{}_s_{}".format(datasetPath.split('/')[-1], scale) + '.hdf5', "features", bufferSize)
dataset.storeClassLabels(le.classes_)

# initialize progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over images
for i in np.arange(0, len(imagePaths), batchSize):
    batchPaths = imagePaths[i:i + batchSize]
    batchLabels = labels[i:i + batchSize]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        # load and resize image
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess image by expanding and subtracting mean RGB value
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # add image to batch
        batchImages.append(image)

    # pass images thr network
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batchSize)
    # reshape features
    features = features.reshape((features.shape[0], 2048 * 7 * 7))

    # add features and labels to dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
