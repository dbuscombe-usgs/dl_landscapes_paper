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

This is a python workflow that requires installation of some third-party libraries. We recommend you use a virtual environment and the conda platform for installing and managing dependencies

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

Details here 


### Using a deep convolutional neural network for semantic segmentation

Details here 











