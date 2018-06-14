# "Landscape classification with deep neural networks" by Daniel Buscombe and Andrew C. Ritchie

The following document contains a workflow to reproduce the results in the above paper. If you find these data and codes useful in your research, please cite!



### Credits

Daniel Buscombe,
School of Earth Sciences & Environmental Sustainability
Northern Arizona University
Flagstaff, AZ
daniel.buscombe@nau.edu


Andrew Ritchie,
Pacific Coastal and Marine Science Center
U.S. Geological Survey
Santa Cruz, CA
aritchie@usgs.gov


## Getting started

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
```


## File structure

The files are organized by data set

### data/
This top-level directory contains the python processing scripts, trained models and associated files


### data/test
This directory contains data associated with model testing


### data/train
This directory contains data associated with model training


## Data sets

### Seabright


### Lake Ontario


### NWPU


### CCRP


### Grand Canyon



## Workflow

### Creating ground truth / label imagery for training/validating a deep neural networks

Details here 

### Retraining a deep convolutional neural network for image recognition

Details here 


### Using a deep convolutional neural network for semantic segmentation

Details here 











