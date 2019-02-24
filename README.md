# Fundamental Matrix Estimation from Multi-View Silhouette Images using Deep Learning

## Why did I create this repository?
I created this repository to share the results of my interdisciplinary project (IDP) which was part of my Master's study at the Technical University Munich (TUM). All the code and documentation in this repo were created as part of this IDP.

## What is this project about?
In this project a deep learning model is trained to estimate fundamental matrices for pairs of silhouette images. The model involves a GAN which is trained during predication time to learn how to represent the second input image given the first one. Then the internal weights of the generator are used as input to a trained regressor network that predicts the fundamental matrix. For details see the documentation pdf.

## What does this repository contain
The repository contains the IDP project of:
* Python source code for dataset generation and loading: folder "dataset_utils"
* Python source code for training and testing of the model: folder "train" as well as the files "train_model.py" to train the model and "test_model.py" to test the model
* Python source code for analyzing the results of the model as well as json files containing the training and testing results: folder "results"
* Documentation of the project including an analysis of the results: file "IDP_Documentation.pdf"
* requirements.txt containing the libs I used

It does neither contain the datasets nor the trained model weights. For creation of the datasets see next section in this README file. If you are also interested in the model weights, then I can provide them.

### Datasets
The datasets for training the model can be created automatically using the provided Python scripts in the "dataset_utils" folder. For this use the file "dataset_preparation.ipynb" in that folder. For some of the datasets the data required for their creation is downloaded automatically. But some datasets (the synthetic datasets) require to manually download 3D-models (.ply-files) and add them to the folder "backup/3d_models" (from the root folder of the project). Some 3D-models which I used can for example be downloaded from the following pages:

http://graphics.im.ntu.edu.tw/~robin/courses/cg03/model/
 * bunny.ply
 * horse.ply
 
https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html
 * airplane.ply
 * bid_dodge.ply
 * chopper.ply
 * galleon.ply
 * stratocaster.ply
 * street_lamp.ply
