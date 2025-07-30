from auxiliar_NN import MLP, assign_weights_to_model, init_weights
from dislib.data.tensor import tensor_from_ds_array 
import torch
import dislib as ds
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dislib.preprocessing import MinMaxScaler
import math
from dislib.pytorch import EncapsulatedFunctionsDistributedPytorch
from dislib.data.array import Array
from dislib.data.tensor import Tensor
import sys
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    INOUT, IN, COLLECTION_OUT, COLLECTION_IN
from pycompss.api.task import task
from dislib.data.tensor import load_dataset

from pycompss.api.api import compss_wait_on
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


def read_dataset(dataset_file, partitions=10):
    data = pd.read_csv(dataset_file)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    x_data = data.loc[:, data.columns != 'Intensity Value 3s']
    y_data = data.loc[:, data.columns == 'Intensity Value 3s']
    x_array = ds.array(x_data, block_size=(math.ceil(x_data.shape[0]/partitions), x_data.shape[1]))
    y_array = ds.array(y_data, block_size=(math.ceil(x_data.shape[0]/partitions), 1))
    return x_array, y_array


def load_data(all_dataset, train_dataset, test_dataset):
    Data_X_arr_All, Data_Y_arr_All = read_dataset(all_dataset, partitions=20)
    x_test, y_test = read_dataset(test_dataset, partitions=4)
    x_train, y_train = read_dataset(train_dataset, partitions=4)
    minmax_scaler_total_x = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler_total_y = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler_total_x.fit(Data_X_arr_All)
    minmax_scaler_total_y.fit(Data_Y_arr_All)
    x_train = minmax_scaler_total_x.transform(x_train)
    y_train = minmax_scaler_total_y.transform(y_train)
    x_test = minmax_scaler_total_x.transform(x_test)
    x_train = x_train.collect()
    y_train = y_train.collect()
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train.reshape(-1, 1))
    x_train = ds.from_pt_tensor(x_train, shape=(8, 1))
    y_train = ds.from_pt_tensor(y_train, shape=(8, 1))
    x_test = x_test.collect()
    y_test = y_test.collect()
    y_test = torch.from_numpy(y_test.reshape(-1, 1))
    y_test = ds.from_pt_tensor(y_test, shape=(8, 1))
    x_test = torch.from_numpy(x_test)
    x_test = ds.from_pt_tensor(x_test, shape=(8, 1))
    return x_train, y_train, x_test, y_test, minmax_scaler_total_x, minmax_scaler_total_y


def train_main_network(x_train, y_train):
    encaps_function = EncapsulatedFunctionsDistributedPytorch(num_workers=4)
    torch_model = MLP()
    torch_model.apply(init_weights)
    criterion = nn.MSELoss
    optimizer = optim.Adam
    optimizer_parameters = {"lr": 0.0002}
    encaps_function.build(torch_model, optimizer, criterion, optimizer_parameters, num_gpu=4, num_nodes=1)
    start_time = time.time()
    trained_weights = encaps_function.fit_synchronous_every_n_epochs_with_GPU(x_train, y_train, 2543, 32, n_epocs_sync=3, shuffle_blocks=False, shuffle_block_data=True)
    training_time = time.time() - start_time
    torch_model = assign_weights_to_model(torch_model, trained_weights)
    return torch_model, x_train, y_train, training_time


def evaluate_main_network(x_test, y_test, torch_model):
    outputs = []
    x_test = x_test.collect()
    torch_model = torch_model.to("cuda:0")
    for x_out_tens in x_test:
        for x_in_tens in x_out_tens:
            output = torch_model(x_in_tens.float().to("cuda:0"))
            outputs.append(output)
    outputs = torch.cat(outputs)
    outputs = outputs.detach().cpu().numpy()
    outputs = ds.array(outputs, block_size=(math.ceil(outputs.shape[0]/8), outputs.shape[1]))
    outputs = minmax_scaler_total_y.inverse_transform(outputs)
    y_test = torch.cat([tens for tensor in y_test.collect() for tens in tensor])
    y_test = ds.array(y_test, block_size=(500000, 1))
    y_test = y_test.collect()
    outputs = outputs.collect()
    print("MSE: " + str(mean_squared_error(y_test, outputs)))
    print("MAE: " + str(mean_absolute_error(y_test, outputs)))
    print("R2 score: " + str(r2_score(y_test, outputs)))
    print("Pearson Corr: " + str(np.corrcoef(y_test, outputs)))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("It's required to specify three paths for the dataset.")
    x_train, y_train, x_test, y_test, minmax_scaler_total_x, minmax_scaler_total_y = load_data(sys.argv[1], 
            sys.argv[2], 
            sys.argv[3])

    model_path = "./weights/mlp_mnist.pth"
    # Original model timing
    num_epochs = 4
    # Get smaller model
    torch_model, x_train, y_train, training_time = train_main_network(x_train, y_train)

    train_data = []
    test_data = []
    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

