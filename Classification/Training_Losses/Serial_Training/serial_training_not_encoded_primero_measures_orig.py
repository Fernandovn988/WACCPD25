from NFNet import NFNet
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




class RandomTranslate:
    def __init__(self, translate=(0.1, 0.1), interpolation=Trans_F.InterpolationMode.BILINEAR):
        """
        translate: (max_horizontal, max_vertical) como fracción del ancho/alto (ej. 0.1 = 10%).
        """
        self.translate = translate
        self.interpolation = interpolation

    def __call__(self, x):
        # x: tensor [B, C, H, W]
        B, C, H, W = x.shape
        max_tx = self.translate[0] * W  # Traslación horizontal máxima en píxeles
        max_ty = self.translate[1] * H  # Traslación vertical máxima en píxeles

        # Genera offsets aleatorios para todo el batch
        tx = torch.empty(B).uniform_(-max_tx, max_tx).to(x.device)
        ty = torch.empty(B).uniform_(-max_ty, max_ty).to(x.device)

        # Matriz de transformación afín (2x3) para cada imagen en el batch
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = 1  # Escala horizontal (sin cambio)
        theta[:, 1, 1] = 1  # Escala vertical (sin cambio)
        theta[:, 0, 2] = tx / (W // 2)  # Normalizado a [-1, 1] (coord. grid_sample)
        theta[:, 1, 2] = ty / (H // 2)

        # Grid y sampling
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class RandomRotation:
    def __init__(self, degrees, interpolation=Trans_F.InterpolationMode.BILINEAR):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.interpolation = interpolation

    def __call__(self, x):
        # x: tensor de forma [B, C, H, W]
        angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        return rotate(x, angle, interpolation=self.interpolation)


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


def assign_weights_to_model(model, trained_weights):
    j=0
    if hasattr(model, 'neural_network_layers'):
        len_nn = len(model.neural_network_layers)
        for i in range(len_nn):
            if hasattr(model.neural_network_layers[i], 'weight'):
                model.neural_network_layers[i].weight = nn.Parameter(trained_weights.neural_network_layers[i].weight)
                j += 1
                model.neural_network_layers[i].bias = nn.Parameter(trained_weights.neural_network_layers[i].bias)
                j += 1
    if hasattr(model, 'dense_neural_network_layers'):
        len_nn = len(model.dense_neural_network_layers)
        aux_j = 0
        for i in range(len_nn):
            if hasattr(model.dense_neural_network_layers[i], 'weight'):
                model.dense_neural_network_layers[i].weight = nn.Parameter(trained_weights.dense_neural_network_layers[i].weight)
                aux_j += 1
                model.dense_neural_network_layers[i].bias = nn.Parameter(trained_weights.dense_neural_network_layers[i].bias)
                aux_j += 1
    return model

def load_data(x_train, y_train, x_test, y_test):
    x_train = torch.load(x_train)
    y_train = torch.load(y_train)
    x_test = torch.load(x_test)
    y_test = torch.load(y_test)
    y_test = ds.from_pt_tensor(y_test, shape=(8, 1))
    x_test = ds.from_pt_tensor(x_test, shape=(8, 1))
    return x_train, y_train, x_test, y_test


def compute_accuracy(torch_model, x_test, y_test_local, y_test_one_hot_encoded):
    torch_model = torch_model.eval().to("cuda:0")
    indexes = 64
    num_batches = math.ceil(x_test[0][0].shape[0]/indexes)
    outputs = []
    for x_out_tens in x_test:
        for x_in_tens in x_out_tens:
            x_in_tens = x_in_tens.to("cuda:0")
            for idx in range(num_batches):
                with torch.no_grad():
                    output = torch_model(x_in_tens[idx * indexes: (idx + 1) *indexes].float())
                    output_cpu = output.cpu()
                    outputs.append(output_cpu)
                    del output
                torch.cuda.empty_cache()
            x_in_tens = x_in_tens.to("cpu")
            del x_in_tens
            torch.cuda.empty_cache()
    outputs = torch.cat(outputs)
    loss = nn.CrossEntropyLoss()
    acc_outputs = process_outputs(outputs)
    acc_outputs = acc_outputs.detach().cpu().numpy()
    return loss(outputs, y_test_one_hot_encoded).item(), accuracy_score(y_test_local, acc_outputs)
    validation_loss.append(loss(outputs, y_test_one_hot_encoded).item())
    outputs = process_outputs(outputs)
    outputs = outputs.detach().cpu().numpy()
    validation_acc.append(accuracy_score(y_test_local, outputs))


def train_main_network(x_train, y_train, x_test, y_test):
    encaps_function = EncapsulatedFunctionsDistributedPytorch(num_workers=4)
    torch_model = NFNet().to("cuda:0")
    torch_model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    indexes=128
    num_batches = math.ceil(x_train.shape[0]/indexes)
    start_time = time.time()
    y_train = y_train.to("cuda:0")
    y_train_cpu = y_train.to("cpu")
    x_train = x_train.float().to("cuda:0")
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []
    x_test = x_test.collect()
    y_test_local = torch.cat([tens for tensor in y_test.collect() for tens in tensor])
    _, y_test_one_hot_encoded = torch.max(y_test_local, dim=1)
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
            epoch_loss.append(loss.item())
            output_train.append(outputs.to("cpu"))
        scheduler.step()
        indexes_s = torch.randperm(len(x_train))
        x_train = x_train[indexes_s]
        y_train = y_train[indexes_s]
        val_loss, val_acc = compute_accuracy(torch_model, x_test, y_test_local, y_test_one_hot_encoded)
        train_loss.append(np.array(epoch_loss).mean())
        output_train = torch.cat(output_train)
        _, output_train = torch.max(output_train, dim=1)
        train_acc.append(accuracy_score(y_train_cpu.detach().numpy(), output_train.detach().numpy()))
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
    training_time = time.time() - start_time
    print("Saving training loss and validation loss")
    df = pd.DataFrame([train_loss])
    df.to_csv("train_loss_1.csv", index=False, decimal=",")
    df = pd.DataFrame([validation_loss])
    df.to_csv("validation_loss_1.csv", index=False, decimal=",")
    df = pd.DataFrame([train_acc])
    df.to_csv("train_acc_1.csv", index=False, decimal=",")
    df = pd.DataFrame([validation_acc])
    df.to_csv("validation_acc_1.csv", index=False, decimal=",")
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
            sys.argv[2], sys.argv[3], sys.argv[4])

    # Original model timing
    # Get smaller model
    torch_model, training_time = train_main_network(x_train, y_train, x_test, y_test)

    print("Evaluate Original Accuracy, MSE or MAE", flush=True)
    print("Time used to train NN: " + str(training_time))
    evaluate_main_network(x_test, y_test, torch_model)

