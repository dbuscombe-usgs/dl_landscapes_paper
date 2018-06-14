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

### Create a python virtual environment usin conda

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

## Creating ground truth / label imagery for training/validating a deep neural networks

Details here 

## Retraining a deep convolutional neural network for image recognition

Details here 


## Using a deep convolutional neural network for semantic segmentation

Details here 











