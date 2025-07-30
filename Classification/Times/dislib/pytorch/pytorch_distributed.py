import os
import time
import copy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from pycompss.api.task import task
from pycompss.api.parameter import IN, IN_DELETE, COMMUTATIVE
from pycompss.api.constraint import constraint
import torch.nn as nn
import torch
import numpy as np
import math

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import os
from sklearn.metrics import accuracy_score
from NFNet import TensorTransformations
import torchvision.transforms.functional as Trans_F

import torch.nn.functional as F
from torchvision import transforms

from torchvision.transforms.functional import rotate, affine


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


def adaptive_gradient_clipping(model, clip_factor=0.01):
    """Adaptive Gradient Clipping (AGC)"""
    #print("WHAT IS MODEL PARAM")
    #print(model.parameters(), flush=True)
    for param in model.parameters():
        #print("PARAM", flush=True)
        #print(param, flush=True)
        if param.grad is not None:
            param_norm = torch.norm(param.detach())
            grad_norm = torch.norm(param.grad.detach())
            max_norm = clip_factor * param_norm
            if grad_norm > max_norm:
                param.grad.mul_(max_norm / (grad_norm + 1e-6))


class PytorchDistributed(object):
    """
    PytorchDistributed object. It is in charge of executing in parallel the
    small trainings inside each epoch of the main training.
    """
    def __init__(self):
        self.model = None
        self.loss = None
        self.optimizer = None

    def build(self, net, loss, optimizer, optimizer_parameters, scheduler=None):
        """
        Sets all the needed parameters and objects for this object.

        Parameters
        ----------
        net:
            Network that will be used during the training
        loss:
            Loss used during the network training
        optimizer:
            Optimizer object used to updated the weights of the network
            during the training.
        optimizer_parameters:
            Parameters to set in the Optimizer
        """
        local_net = net
        self.model = local_net
        self.loss = loss()
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters
        if scheduler is not None:
            self.scheduler = copy.deepcopy(scheduler)
        else:
            self.scheduler = None

    def train_cnn_batch_GPU(self, model_parameters, x_train,
                            y_train, num_batches, shuffle_block_data):
        """
        Performs a training of one of the workers network
        on the corresponding part of the data.

        Parameters
        ----------
        model_parameters: tensor
            Weights and biases of the different layers of the network
            that will be used in this small training.
        x_train: torch.tensor
            Samples of the training data
        y_train: torch.tensor
            Labels, regression values or etc. of training data.
        num_batches: int
            Number of batches that will be done
        shuffle_block_data: boolean
            Whether to shuffle or not the training data used.

        Returns
        -------
        model_parameters: tensor
            Updated weights and biases of the different layers of the network
            after the training.
        """
        return self._train_cnn_batch_GPU(model_parameters, x_train,
                                         y_train, num_batches,
                                         shuffle_block_data)

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task(target_direction=IN, model_parameters=IN_DELETE, x_train=IN, y_train=IN, num_batches=IN, shuffle_block_data=IN)
    def _train_cnn_batch_GPU(self, model_parameters, x_train,
                             y_train, num_batches, shuffle_block_data):
        if shuffle_block_data:
            idx = torch.randperm(x_train.shape[0])
            if not isinstance(x_train.size, int):
                x_train = x_train[idx].view(x_train.size())
            else:
                if not torch.is_tensor(x_train):
                    x_train = x_train[idx]
                else:
                    x_train = torch.from_numpy(x_train)
                    x_train = x_train[idx].view(x_train.size())
            if not isinstance(y_train.size, int):
                if len(y_train.size()) > 1:
                    y_train = y_train[idx].view(y_train.size())
                else:
                    y_train = y_train[idx]
            else:
                if not torch.is_tensor(y_train):
                    y_train = y_train[idx]
                else:
                    y_train = torch.from_numpy(y_train)
                    y_train = y_train[idx].view(y_train.size())

        if hasattr(self.model, 'neural_network_layers'):
            len_nn = len(self.model.neural_network_layers)
            for i in range(len_nn):
                if hasattr(model_parameters.neural_network_layers[i],
                           'weight'):
                    self.model.neural_network_layers[i].weight = \
                        nn.Parameter(
                            model_parameters.neural_network_layers[i].
                            weight.float())
                if hasattr(model_parameters.neural_network_layers[i],
                           'bias'):
                    self.model.neural_network_layers[i].bias = \
                        nn.Parameter(
                            model_parameters.neural_network_layers[i].bias.
                            float())
                if hasattr(self.model.neural_network_layers[i],
                       'shortcut'):
                    len_shortcut = len(self.model.neural_network_layers[i].shortcut)
                    for k in range(len_shortcut):
                        if hasattr(self.model.neural_network_layers[i].shortcut[k],
                               'weight'):
                            self.model.neural_network_layers[i].shortcut[k].weight = \
                                nn.Parameter(model_parameters.neural_network_layers[i].shortcut[k].
                                weight.float())
                            self.model.neural_network_layers[i].shortcut[k].bias = \
                                nn.Parameter(model_parameters.neural_network_layers[i].shortcut[k].
                                bias.float())
                        if hasattr(self.model.neural_network_layers[i].shortcut[k],
                               'alpha'):
                            self.model.neural_network_layers[i].shortcut[k].alpha = \
                                nn.Parameter(model_parameters.neural_network_layers[i].shortcut[k].alpha)
                if hasattr(self.model.neural_network_layers[i],
                           'layers'):
                    len_layers = len(self.model.neural_network_layers[i].layers)
                    for k in range(len_layers):
                        if hasattr(self.model.neural_network_layers[i].layers[k],
                               'weight'):
                            self.model.neural_network_layers[i].layers[k].weight = \
                                nn.Parameter(model_parameters.neural_network_layers[i].
                                layers[k].weight.float())
                            self.model.neural_network_layers[i].layers[k].bias = \
                                nn.Parameter(model_parameters.neural_network_layers[i].
                                layers[k].bias.float())
                        if hasattr(self.model.neural_network_layers[i].layers[k],
                               'alpha'):
                            self.model.neural_network_layers[i].layers[k].alpha = \
                                nn.Parameter(model_parameters.neural_network_layers[i].layers[k].alpha)
        if hasattr(self.model, 'dense_neural_network_layers'):
            len_nn = len(model_parameters.dense_neural_network_layers)
            for i in range(len_nn):
                if hasattr(
                        model_parameters.dense_neural_network_layers[i],
                        'weight'):
                    self.model.dense_neural_network_layers[i].weight = \
                        nn.Parameter(
                            model_parameters.dense_neural_network_layers[i].
                            weight.float())
                if hasattr(model_parameters.dense_neural_network_layers[i],
                           'bias'):
                    self.model.dense_neural_network_layers[i].bias = \
                        nn.Parameter(
                            model_parameters.dense_neural_network_layers[i].
                            bias.float())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = self.optimizer(self.model.parameters(),
                                   **self.optimizer_parameters)
        x_train = x_train.float().to(device)
        #if len(y_train.shape) == 1:
        #    y_train = y_train.float()
        #elif y_train.shape[-1] == 1:
        #    y_train = y_train.float()
        indexes = math.ceil(x_train.shape[0] / num_batches)
        losses_epoch = []
        if isinstance(self.loss, nn.CrossEntropyLoss):
            y_train = y_train.long()
        true_labels = y_train.to(device)
        output_list = []
        for idx in range(num_batches):
            optimizer.zero_grad()
            inputs = x_train[idx*indexes:(idx+1)*indexes]
            inputs = TensorTransformations.random_crop(inputs, padding=4)
            inputs = TensorTransformations.random_horizontal_flip(inputs)
            outputs = self.model(inputs)
            if len(true_labels.shape) > 1:
                loss = self.loss(outputs,
                                 true_labels[idx*indexes:(idx+1)*indexes])
            else:
                loss = self.loss(outputs,
                                 true_labels[idx*indexes:(idx+1)*indexes])
            loss.backward()
            adaptive_gradient_clipping(self.model, clip_factor=0.01)
            optimizer.step()
            losses_epoch.append(loss.item())
            output_list.append(outputs.to("cpu"))
        output_list = torch.cat(output_list)
        _, output_list = torch.max(output_list, dim=1)
        true_labels = true_labels.to("cpu")
        self.model = self.model.to("cpu")
        return self.model

    def aggregate_parameters_async(self,
                                   model_params,
                                   parameters_to_aggregate):
        """
        Function that aggregates in commutative and without requiring a
        synchronization the weights of the network
        generated by the different trainings.

        Parameters
        ----------
        model_params: tensor
            Weights and biases of the different layers of the main
            network.
        parameters_to_aggregate:
            Weights and biases generated through the training

        Returns
        -------
        model_params: tensor
            Updated weights and biases of the different layers of the main
            network.
        """
        return self._aggregate_parameters_async(model_params,
                                                parameters_to_aggregate)

    @constraint(processors=[
        {'processorType': 'CPU', 'computingUnits': '${ComputingUnits}'},
        {'processorType': 'GPU', 'computingUnits': '${ComputingUnitsGPUs}'}])
    @task(model_params=COMMUTATIVE, parameters_to_aggregate=IN,
          target_direction=IN)
    def _aggregate_parameters_async(self, model_params,
                                    parameters_to_aggregate):
        final_weights = []
        worker_weights = []
        for param in model_params.parameters():
            worker_weights.append(param)
        final_weights.append(worker_weights)
        worker_weights = []
        for param in parameters_to_aggregate.parameters():
            worker_weights.append(param.to("cuda:0"))
        final_weights.append(worker_weights)
        final_added_parameters = final_weights[0]
        for i in range(len(final_weights[0])):
            for j in range(1, len(final_weights)):
                final_added_parameters[i] = final_added_parameters[i] + \
                                            final_weights[j][i]
            final_added_parameters[i] = final_added_parameters[i]
        for i in range(len(final_weights[0])):
            final_added_parameters[i] = final_added_parameters[i] / 2
        j = 0
        if hasattr(model_params, 'neural_network_layers'):
            len_nn = len(model_params.neural_network_layers)
            for i in range(len_nn):
                if hasattr(model_params.neural_network_layers[i], 'weight'):
                    model_params.neural_network_layers[i].weight = \
                        nn.Parameter(final_added_parameters[j].float())
                    j += 1
                    model_params.neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[j].float())
                    j += 1
                if hasattr(model_params.neural_network_layers[i],
                       'shortcut'):
                    len_shortcut = len(model_params.neural_network_layers[i].shortcut)
                    for k in range(len_shortcut):
                        if hasattr(model_params.neural_network_layers[i].shortcut[k],
                               'weight'):
                            model_params.neural_network_layers[i].shortcut[k].weight = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
                            model_params.neural_network_layers[i].shortcut[k].bias = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
                        if hasattr(model_params.neural_network_layers[i].shortcut[k],
                               'alpha'):
                            model_params.neural_network_layers[i].shortcut[k].alpha = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
                if hasattr(model_params.neural_network_layers[i],
                           'layers'):
                    len_layers = len(model_params.neural_network_layers[i].layers)
                    for k in range(len_layers):
                        if hasattr(model_params.neural_network_layers[i].layers[k],
                               'weight'):
                            model_params.neural_network_layers[i].layers[k].weight = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
                            model_params.neural_network_layers[i].layers[k].bias = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
                        if hasattr(model_params.neural_network_layers[i].layers[k],
                               'alpha'):
                            model_params.neural_network_layers[i].layers[k].alpha = \
                                nn.Parameter(final_added_parameters[j].float())
                            j += 1
        if hasattr(model_params, 'dense_neural_network_layers'):
            len_nn = len(model_params.dense_neural_network_layers)
            aux_j = 0
            for i in range(len_nn):
                if hasattr(model_params.dense_neural_network_layers[i],
                           'weight'):
                    model_params.dense_neural_network_layers[i].weight = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
                    model_params.dense_neural_network_layers[i].bias = \
                        nn.Parameter(final_added_parameters[aux_j + j].float())
                    aux_j += 1
        return model_params

