# DeepTrajectory

<img align="right" width="250" height="475" src="/img/method.png?raw=true">

DeepTrajectory is a deep recurrent neural network based on gated recurrent units to identify improved conformational states from refinement trajectory data in order to assist accurate protein structure prediction.

This repository contains the source code of the model together with helper functions to measure the performance during training and validation.

## Data-set availability

The data-set for training and testing can be downloaded from https://zenodo.org/record/1183354. This webpage contains links to the raw PDB files of all trajectories used in this work, the feature table in CSV format and the cross-validation assignment as a CSV file. 


## Usage

The following python dependencies are required to run the code: tensorflow (version 1.0.0), scikit-learn, numpy, pandas.

The ipython notebook in src/training_example.ipynb shows an example how to train the model. Please make sure that you have downloaded and extracted the CSV data to the sub folder data/.
