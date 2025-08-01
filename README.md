# WACCPD25
This GitHub repository containes the Artifact corresponding to the article: "Scalable Neural Network Training: Distributed Data-Parallel Approaches", submitted to the WACCPD 2025 workshop.

This GitHub repository is divided into two different folders: **Classification** and **Regression**. The experiments evaluate the training times and results obtained by the neural networks on two different problems, one is a classification problem and the other regression. Each folder contains the corresponding experiments.

## Requirements

In order to reproduce the experiments some requirements should be met.

### Hardware

It is required to have access to a large number of GPUs, some of the experiments require up to 64 GPUs. The GPUs used are Nvidia Hopper with 64 HBM2 memory. However, other GPUs can be used; of course, it can't be expected to obtain the same times using other hardware, maybe a similar time, but different.

### Software

The Python version used to execute the experiments was Python '3.12.1'. One software package required is PyTorch, version '2.3.0+cu121'. This can be installed using the command:

$ python3 -m pip install torch==2.3.0+cu121

Finally, the last software requirement is PyCOMPSs version '3.3.3'. The installation steps are clearly in [COMPSs documentation page](https://compss-doc.readthedocs.io/en/stable/Sections/01_Installation.html)

### Data

The dataset used in the Classification experiments is publicly available. It has a [GitHub](https://github.com/BayesWatch/cinic-10), where it is explained how to load it, its distribution, etc and a link to downloaded the dataset is available. It can also be downloaded from [Kaggle](https://www.kaggle.com/datasets/mengcius/cinic10/data).

## Execution

In order to execute any of the experiments once the requirements are installed it is only needed to execute the corresponding bash script specifying the paths to the data. For example with the classification problem:

$ ./launch_nn_double.sh $PATH_TO_X_TRAIN $PATH_TO_Y_TRAIN $PATH_TO_X_TEST $PATH_TO_Y_TEST

## Classification

This folder contains the classification experiments that are shown in the article.

## Regression

This folder contains the regression experiments shown in the article. It corresponds to the federated training part of the article.

