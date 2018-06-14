# "Landscape classification with deep neural networks" by Daniel Buscombe and Andrew C. Ritchie

The following document contains a workflow to reproduce the results in the above paper. If you find these data and codes useful in your research, please cite!

## Create a python virtual environment usin conda

```
conda create --name tfpy35 python=3.5
conda activate tfpy35
```

## Install the pydensecrf package from conda forge

```
conda config --add channels conda-forge
conda install pydensecrf
conda config --remove channels conda-forge
```

## Install other python libraries

```
pip install Cython
pip install numpy scipy matplotlib scikit-image scikit-learn
pip install joblib
pip install tensorflow tensorflow_hub
```


