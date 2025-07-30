# Classification
Inside this folder one can find the experiments included in the paper: "Scalable Neural Network Training: Distributed Data-Parallel Approaches". 
Inside the **Figures** folder one can find the training curves obtained using different training strategies. The folder **Generate_Dataset** contains the script used to load the CINIC-10 dataset and transform it 
into PyTorch tensors that were used in the experiments. 

There are two more folders: **Training_Losses** and **Times**

## Generate Dataset

This folder contains the script used to generate the dataset of the experiments. We used both validation and training data as training data. We used the test data to measure the reported accuracies of the different experiments. In order to create the dataset it is required to execute the script like:

$ python3 transform_dataset.py $PATH_TO_CINIC10 $PATH_WHERE_STORE_TRAIN_TEST_DATA

## Training Losses
The experiments contained in this folder try to replicate the figures containing the training losses contained inside the folder **Figures** and in the article. The scripts inside **./Training_Losses/Asynchronous_Training/** are the ones used to obtain the data that is shown in Figure 2 in the article. 
In order to reproduce any of these experiments one should go inside the corresponding folder, f. example:

$ cd ./Training_Losses/Asynchronous_Training/4_workers

Then, specify the path to the corresponding dislib, in this case the dislib contained in ./Training_Losses/Asynchronous_Training:

$ export PATH_TO_DISLIB=....

Finally, launch the execution using the bash script:

./launch_nn_double.sh $PATH_TO_X_TRAIN $PATH_TO_Y_TRAIN $PATH_TO_X_TEST $PATH_TO_Y_TEST

The serial values used in all the figures were generated using the scripts contained in the folder **./Training_Losses/Serial_Training**. All the figures reflect the mean values of 5 executions and were generated using google sheets figures.

## Times

The experiments contained in this folder correspond to the Figures 3 to 8 and the Table 1. 

The experiments that reproduce the results included in Table 1 are inside the folders: ./Times/Asynchronous_Training ./Times/Synchronous_Training ./Times/Synchronous_Every_2_Epochs_Training ./Times/Serial_Training

The previous folders correspond to the experiments shown in Figure 3, with the addition of the experiments in folder: ./Times/Synchronous_Every_3_Epochs_Training

The folder: **./Times/32_workers_synchronous** contains the scripts to reproduce the results shown in Figure 4 of the article (execution times and speedup up to 32 workers). Then the folder **./Times/64_workers_asynchronous** contains the script to reproduce the values shown in Figure 5. Additionally, these experiments can be used to reproduce what is shown in Figure 8, just by changeng the number of partitions in which the training is loaded in the different executions.

In order to reproduce the values shown in Figure 6, it is required to execute the scripts contained in: **./Times/Pytorch_DDP**, the execution of these scripts will return the times obtained using PyTorch DDP to train the neural network. The times shown in Figure 6 using our strategies correspond to the values obtained using the scripts in folder: **./Times/Asynchronous_Training**

Finally, figure 7 can be replicated with the results obtained executing the scripts inside **./Times/Synchronous_Training**, with the partitions used by default in the training data, and changing the partitions to 16.

