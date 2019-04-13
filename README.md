# Pattern Recognition Project

## Description

This repository contains the Jupyter notebook files used in our project for the pattern recognition course (SYDE 675).

# Classifier Testing Summary

The following classifiers will be used for prediction of hospital readmission. The hyperparameters of these classifiers will be tuned using random search with n_iters ~ 1000.

- [ ] Decision Tree
- [ ] Random Forest
- [ ] Ada Boost
- [ ] Gradient Boosting
- [ ] Gaussian Process
- [ ] MLP
- [ ] kNN
- [ ] SVM

## Set Up
In order to access the notebook files you can follow these steps to get an environment setup with the necessary packages.
1. Install the following prerequistes:
    - Anaconda 5.3 with Python 3.7 - [Link Here](https://www.anaconda.com/download/)
2. Use Anaconda to install packages in a new or existing virtual environment:
    `conda create -n new-env --file requirements.txt`
    or
    `conda install --yes --file requirements.txt`
3. Setup nbdime for a better diff tool with jupyter notebooks:
    `nbdime config-git --enable --global`
4. Activate the created environment if you are using one.
    `activate new-env`
5. Run Jupyter Lab to access the notebooks.
    `jupyter lab`
