# "Landscape classification with deep neural networks" by Daniel Buscombe and Andrew C. Ritchie

An EarthArXiv preprint of this manuscript may be found [here](https://eartharxiv.org/5mx3c)

The following document contains a workflow to reproduce the results in the above paper. If you find these data and codes useful in your research, please cite!


### Credits

Daniel [Buscombe](https://www.danielbuscombe.com/),
School of Earth Sciences & Environmental Sustainability
Northern Arizona University
Flagstaff, AZ
daniel.buscombe@nau.edu

and

Andrew Ritchie,
Pacific Coastal and Marine Science Center
U.S. Geological Survey
Santa Cruz, CA
aritchie@usgs.gov


## Getting started

This is a python workflow that requires installation of some third-party libraries. We recommend you use a virtual environment and the conda platform for installing and managing dependencies. Reproducing the results in the paper will require some experience and familiarity with running and modifying python scripts 

### Create a python virtual environment using conda

```
conda create --name tfpy35 python=3.5
conda activate tfpy35
```

### Install the pydensecrf package from conda forge

```
conda config --add channels conda-forge
conda install pydensecrf
conda config --remove channels conda-forge
```

### Install other python libraries

```
pip install Cython
pip install numpy scipy matplotlib scikit-image scikit-learn
pip install joblib
pip install tensorflow tensorflow_hub
conda install opencv
```


## Data sets

### Seabright
The dataset consists of 13 images of the shorefront at Seabright, Santa Cruz, CA. Images were collected from a fixed-wing aircraft in February 2016, of which a random subset of seven were used for training, and six for testing. Training and testing tiles were generated for seven classes, namely water, vegetation, sand, road, foam/surf, anthropogenic, and other natural terrain.

### Lake Ontario
The dataset consists of 48 images obtained in July 2017 from a Ricoh GRII camera mounted to a 3DR Solo quadcopter, a small unmanned aerial system (UAS), flying 80-100 meters above ground level in the vicinity of Braddock Bay, New York, on the shores of southern Lake Ontario [64]. A random subset of 24 were used for training, and 24 for testing. Training and testing tiles were generated for five classes, namely water, sediment, vegetation, other natural terrain, and anthropogenic. 

### NWPU
NWPU-RESISC45 is a publicly available benchmark for REmote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). We chose to use a subset of 11 classes corresponding to natural landforms and land cover, namely: beach, chaparral, desert, forest, island, lake, meadow, mountain, river, sea ice, and wetland. All images are 256x256 pixels. We randomly chose 350 images from each class for DCNN training, and 350 for testing.

### CCRP
The dataset consists of a sample of 75 images from the California Coastal Records Project (CCRP), of which 45 were used for training, and 30 for testing. Training and testing tiles were generated for ten classes, namely: sand, cliff, other terrain, vegetation, water, foam/surf, swash, sky, road, and other anthropogenic. 

### Grand Canyon
The dataset consists of 14 images collected from a stationary autonomous camera systems monitoring eddy sandbars along the Colorado River in Grand Canyon. One image from each of seven sites were used for training, and one from each those of same seven sites were used for testing. Training and testing tiles were generated for four classes: water, sand, vegetation, and rock/scree/other terrain. Images from the California Coastal Records Project are Copyright (C) 2002-2018 Kenneth & Gabrielle Adelman, www.Californiacoastline.org.


## File structure

The files are organized by data set

### data/
This top-level directory contains the python processing scripts, trained models and associated files


### data/test
This directory contains data associated with model testing


### data/train
This directory contains data associated with model training


## Workflow

### Creating ground truth / label imagery for training/validating a deep neural networks

Usage: 

```
python create_groundtruth.py -i "image to be processed" -w "horizontal window size in pixels"
```

Example usage:

```
python seabright\create_groundtruth.py -i seabright\test\usgs_pcmsc_2016_02_05_223923.TIF-0.jpg -w 500
```

This script works on one image at a time. The parameter 'w' dictates how much of the image is worked on at once. For each window, the class labels are iterated through. The class appears in the title of the window on screen. Using the mouse, the user must provide examples of each class present in the image. The number and extent of these manual annotations will vary depending on scene complexity (more, for more intra-class variation).

Outputs:

1. a .png file showing the input image, manual annotations, and pixel-level predictions
2. a .mat (matlab format) file with the following fields: 'sparse' (these are the manual annotations provided by the user); 'labels' (the label codes used during classification); and 'class' (the CRF-estimated per-pixel labels)

Example python code for loading the .mat files:

```
from scipy.io import loadmat
dat = loadmat('seabright\test\usgs_pcmsc_2016_02_05_223923.TIF-0_ares.mat')
print(dat.keys())
```


### Retraining a deep convolutional neural network for image recognition

This script will retrain a pre-trained DCNN architecture accessed through Tensorflow-Hub

First, you must unzip all the zipped folders in the directories. Training and testing image tiles should be arranged as jpeg images within folders, with the folder name corresponding to the class label. The paranet directory should be called either 'train' or 'test'. Example: ccr\train\beach and ccr\train\swash

Then, the usage is

```
python retrain.py --image_dir "name of directory containing subfolders with image tiles" \
    --tfhub_module "url of model" \
    --how_many_training_steps "number of training steps" --learning_rate "learning rate" --output_labels "output file containing labels" --output_graph "output file containing the retrained model"
```

The full list of available image recognition modules can be found [here](https://www.tensorflow.org/hub/modules/image)

Example usage:

```
python ccr\retrain.py --image_dir train/tile_96 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 1000 --learning_rate 0.01 --output_labels labels.txt --output_graph seabright_mobilenetv2_96_3000_001.pb
```

Or bash or git-bash users might run the provided script, retrain.sh


### Testing image recognition performance

Example usage:

```
python nwpu\semseg_crf.py 
```

This script will search for image tiles under the \test\ folder structure, classify each tile, and then compile statistics about average performance. It will also print a confusion matrix like those that are shown in the manuscript, showing model-observation performance for each class. Mean accuracy, F1 score, and posterior probabilities will be printed to screen. 


### Using a deep convolutional neural network and conditional random field for semantic segmentation

This script carries out semantic segmentation using a CRF where unary potentials are given by the classification outputs from a trained DCNN model in small regions of the image

Example usage:

```
python gc\semseg_crf.py 
```

Tunable parameters:

* tile: tile size. Here, 96 or 224, but that could change if a different DCNN model is retrained
* winprop: proportion of the tile to be assigned with the class predicted by the DCNN (1=whole tile, 0.5=half the tile, etc)
* direc: directory with images to test
* prob_thres: probability threshold for DCNN-estimated classes, values above which are kept for CRF classification. For example, prob_thres=0.5 means that DCNN estimated labels with a posterior probability of less than 0.5 are ignored
* n_iter: number of CRF iterations (larger = more accurate, to a point where accuracy starts to plateau)
* theta: CRF parameters (see paper for description)
* decim: decimation factor. Dictates how many tiles within the image will be classified using the DCNN. For example, decim=2 means half of all tiles will be used, decim=4 means a quarter, etc
* fct: image downsampling factor. This parameter will downsample the image ahead of pixelwise classification with the CRF. It is for faster predictions. For example, fct=0.5 means the image will be downsampled to half size. All final classifications are rescaled so outputs are the same size (number of pixels) as inputs.
* class_file: the text file produced by retrain.py containing the DCNN model classes
* classifier_file: the .pb file produced by retrain.py containing the retrained tensorflow graph

### Testing semantic segmentation performance

Example usage:

```
python ontario\test_semseg.py 
```

This script will search for images under the \test\ folder structure and their associated CNN-CRF semantic segmentations, and then compile statistics about average performance for pixelwise classification. It will also print a confusion matrix like those that are shown in the manuscript, showing model-observation performance for each class. Mean precision, recall, F1 score, and class areas (in pixels) will be printed to screen. 



