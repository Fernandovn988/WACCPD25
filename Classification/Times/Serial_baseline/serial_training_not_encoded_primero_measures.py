from NFNet import NFNet
import torch
import dislib as ds
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
from dislib.data.tensor import Tensor

from pycompss.api.constraint import constraint
from pycompss.api.parameter import Type, Depth, \
    INOUT, IN, COLLECTION_OUT, COLLECTION_IN
from pycompss.api.task import task
from dislib.data.tensor import load_dataset
import sys
from pycompss.api.api import compss_wait_on
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score
import time
import pandas as pd

from sklearn.metrics import accuracy_score
import torchvision.transforms.functional as Trans_F
from torchvision import transforms

import torch.nn.functional as F

from torchvision.transforms.functional import rotate, affine


def adaptive_gradient_clipping(model, clip_factor=0.01):
    """Adaptive Gradient Clipping (AGC)"""
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.detach())
            grad_norm = torch.norm(param.grad.detach())
            max_norm = clip_factor * param_norm
            if grad_norm > max_norm:
                param.grad.mul_(max_norm / (grad_norm + 1e-6))


class TensorTransformations:
    """Aplica transformaciones sobre tensores (C,H,W) en lugar de PIL Images"""

    @staticmethod
    def random_crop(tensor, padding=4):
        """Crop aleatorio con padding (similar a RandomCrop)"""
        if padding > 0:
            padded = Trans_F.pad(tensor, padding, padding_mode='reflect')
        else:
            padded = tensor
        h, w = padded.shape[-2:]
        new_h, new_w = tensor.shape[-2:]
        top = torch.randint(0, h - new_h, (1,)).item()
        left = torch.randint(0, w - new_w, (1,)).item()
        return Trans_F.crop(padded, top, left, new_h, new_w)

    @staticmethod
    def random_horizontal_flip(tensor, p=0.5):
        """Flip horizontal con probabilidad p"""
        if torch.rand(1) < p:
            return Trans_F.hflip(tensor)
        return tensor



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def process_outputs(output_nn):
    _, indices = torch.max(output_nn, dim=1)
    binary_output = torch.zeros_like(output_nn)
    binary_output[torch.arange(output_nn.size(0)), indices] = 1
    return binary_output



def load_data(x_train, y_train, x_test, y_test):
    x_train = torch.load(x_train)
    y_train = torch.load(y_train)
    x_test = torch.load(x_test)
    y_test = torch.load(y_test)
    y_test = ds.data.tensor.from_pt_tensor(y_test, shape=(8, 1))
    x_test = ds.data.tensor.from_pt_tensor(x_test, shape=(8, 1))
    return x_train, y_train, x_test, y_test


def train_main_network(x_train, y_train, x_test, y_test):
    torch_model = NFNet().to("cuda:0")
    torch_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    indexes=128
    num_batches = math.ceil(x_train.shape[0]/indexes)
    start_time = time.time()
    y_train = y_train.to("cuda:0")
    x_train = x_train.float().to("cuda:0")
    start_time = time.time()
    for _ in range(50):
        epoch_loss = []
        output_train = []
        for idx in range(num_batches):
            optimizer.zero_grad()
            batch = x_train[idx*indexes:(idx+1)*indexes]
            batch = TensorTransformations.random_crop(batch)
            batch = TensorTransformations.random_horizontal_flip(batch)
            outputs = torch_model(batch)
            loss = criterion(outputs,
                   y_train[idx*outputs.shape[0]:(idx+1)*outputs.shape[0]])
            loss.backward()
            adaptive_gradient_clipping(torch_model, clip_factor=0.01)
            optimizer.step()
        scheduler.step()
    training_time = time.time() - start_time
    return torch_model, training_time


def evaluate_main_network(x_test, y_test, torch_model):
    outputs = []
    x_test = x_test.collect()
    torch_model = torch_model.eval()
    indexes = 384
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
        raise ValueError("Paths to train and test data should be provided")
    x_train, y_train, x_test, y_test = load_data(sys.argv[1],
            sys.argv[2],
            sys.argv[3], sys.argv[4])

    # Original model timing
    # Get smaller model
    torch_model, training_time = train_main_network(x_train, y_train, x_test, y_test)

    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

