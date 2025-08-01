from NFNet import NFNet, assign_weights_to_model, init_weights
import torch
import dislib as ds
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
from dislib.pytorch.encapsulated_functions_distributed import EncapsulatedFunctionsDistributedPytorch
from dislib.data.tensor import Tensor
import sys
from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    INOUT, IN, COLLECTION_OUT, COLLECTION_IN
from pycompss.api.task import task
from dislib.data.tensor import load_dataset

from pycompss.api.api import compss_wait_on
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv


def process_outputs(output_nn):
    _, indices = torch.max(output_nn, dim=1)
    binary_output = torch.zeros_like(output_nn)
    binary_output[torch.arange(output_nn.size(0)), indices] = 1
    return binary_output


def load_data(x_train, y_train, x_test, y_test):
    x_train_1 = torch.load(x_train).float()
    x_train_2 = torch.load(x_train).float()
    x_train = torch.cat((x_train_1, x_train_2))
    x_train = ds.data.tensor.from_pt_tensor(x_train, shape=(64, 1))
    y_train_1 = torch.load(y_train).float()
    y_train_2 = torch.load(y_train).float()
    y_train = torch.cat((y_train_1, y_train_2))
    y_train = ds.data.tensor.from_pt_tensor(y_train, shape=(64, 1))
    x_test = torch.load(x_test)
    x_test = ds.data.tensor.from_pt_tensor(x_test, shape=(8, 1))
    y_test = torch.load(y_test)
    y_test = ds.data.tensor.from_pt_tensor(y_test, shape=(8, 1))
    torch.cuda.empty_cache()
    return x_train, y_train, x_test, y_test


def train_main_network(x_train, y_train, x_test, y_test):
    encaps_function = EncapsulatedFunctionsDistributedPytorch(num_workers=8)
    torch_model = NFNet().to("cuda:0")
    torch_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss
    optimizer = optim.SGD
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    optimizer_parameters = {"lr": 0.1, "momentum": 0.9, "weight_decay":0.0001}
    indexes=128
    num_batches = math.ceil(x_train.tensor_shape[0]/indexes)
    encaps_function.build(torch_model, optimizer, criterion, optimizer_parameters, scheduler=scheduler, T_max=100, eta_min=0, num_gpu=4, num_nodes=2) 
    start_time = time.time()
    trained_weights = encaps_function.fit_synchronous_with_GPU(x_train, y_train, num_batches, 50) 
    training_time = time.time() - start_time
    torch_model = assign_weights_to_model(torch_model, trained_weights)
    return torch_model, training_time


def evaluate_main_network(x_test, y_test, torch_model):
    outputs = []
    x_test = x_test.collect()
    torch.cuda.empty_cache()
    torch_model = torch_model.eval().to("cuda:0")
    indexes = 128
    num_batches = math.ceil(x_test[0][0].shape[0]/indexes)
    for x_out_tens in x_test:
        for x_in_tens in x_out_tens:
            x_in_tens = x_in_tens.to("cuda:0")
            for idx in range(num_batches):
                with torch.no_grad():
                    output = torch_model(x_in_tens[idx * indexes: (idx + 1) *indexes].float())
                output_cpu = output.cpu()
                outputs.append(output_cpu)
                del output
            x_in_tens = x_in_tens.to("cpu")
            del x_in_tens
            torch.cuda.empty_cache()
    outputs = torch.cat(outputs)
    y_test = torch.cat([tens for tensor in y_test.collect() for tens in tensor])
    y_test = ds.array(y_test, block_size=(15000, 10))
    y_test = y_test.collect()
    outputs = process_outputs(outputs)
    outputs = outputs.detach().cpu().numpy()
    print("Accuracy: " + str(accuracy_score(y_test, outputs)))
    print("Recall: " + str(recall_score(y_test, outputs, average=None)))
    print("Precision: " + str(precision_score(y_test, outputs, average=None)))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("It is required to specify training and testing dataset paths")
    
    x_train, y_train, x_test, y_test = load_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # Original model timing
    # Get smaller model
    torch_model, training_time = train_main_network(x_train, y_train, x_test, y_test)

    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

