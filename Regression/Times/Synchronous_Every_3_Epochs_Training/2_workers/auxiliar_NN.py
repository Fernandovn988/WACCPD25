from pycompss.api.task import task
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import dislib as ds


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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_neural_network_layers = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.dense_neural_network_layers(x)
        return logits

