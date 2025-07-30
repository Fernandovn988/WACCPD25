import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


import torchvision.transforms.functional as Trans_F
from torchvision import transforms
from PIL import Image

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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
            if hasattr(model.neural_network_layers[i], 'shortcut'):
                len_shortcut = len(model.neural_network_layers[i].shortcut)
                for k in range(len_shortcut):
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                           'weight'):
                        model.neural_network_layers[i].shortcut[k].weight = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].weight)
                        j += 1
                        model.neural_network_layers[i].shortcut[k].bias = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].shortcut[k],
                           'alpha'):
                        model.neural_network_layers[i].shortcut[k].alpha = \
                            nn.Parameter(trained_weights.neural_network_layers[i].shortcut[k].alpha)
                        j += 1
            if hasattr(model.neural_network_layers[i],
                       'layers'):
                len_layers = len(model.neural_network_layers[i].layers)
                for k in range(len_layers):
                    if hasattr(model.neural_network_layers[i].layers[k],
                           'weight'):
                        model.neural_network_layers[i].layers[k].weight = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].weight)
                        j += 1
                        model.neural_network_layers[i].layers[k].bias = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].bias)
                        j += 1
                    if hasattr(model.neural_network_layers[i].layers[k],
                           'alpha'):
                        model.neural_network_layers[i].layers[k].alpha = \
                            nn.Parameter(trained_weights.neural_network_layers[i].layers[k].alpha)
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

    @staticmethod
    def normalize(tensor, mean=[0.478, 0.503, 0.467], std=[0.237, 0.230, 0.234]):
        """Normalización con stats de CINIC-10"""
        return Trans_F.normalize(tensor, mean, std)

# 2. Data Augmentation avanzada (para tensores)
class AdvancedAugmentations:
    """Implementaciones eficientes para tensores"""

    @staticmethod
    def cutmix(batch, alpha=1.0):
        """CutMix: Mezcla regiones entre imágenes del batch"""
        indices = torch.randperm(batch.size(0))
        shuffled = batch[indices]
        lam = np.random.beta(alpha, alpha)

        # Generar máscara binaria
        h, w = batch.shape[-2:]
        rx, ry = torch.randint(w, (1,)).item(), torch.randint(h, (1,)).item()
        rw, rh = int(w * torch.sqrt(1 - lam)), int(h * torch.sqrt(1 - lam))
        x1, y1 = max(0, rx - rw // 2), max(0, ry - rh // 2)
        x2, y2 = min(w, x1 + rw), min(h, y1 + rh)

        # Aplicar mezcla
        batch[:, :, y1:y2, x1:x2] = shuffled[:, :, y1:y2, x1:x2]
        return batch

    @staticmethod
    def mixup(batch, targets, alpha=0.4):
        """MixUp: Interpolación lineal entre imágenes"""
        indices = torch.randperm(batch.size(0))
        shuffled_batch = batch[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)
        batch = lam * batch + (1 - lam) * shuffled_batch
        return batch, targets, shuffled_targets, lam



class ScaledStdConv2d(nn.Conv2d):
    """Capa convolucional con Scaled Weight Standardization (sin BN)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gamma=1.0, eps=1e-6):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.gamma = gamma
        self.eps = eps
        #nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.zeros_(self.bias)  # Biases a cero
        
    def forward(self, x):
        # Weight Standardization
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True)
        weight = (weight - mean) / (var + self.eps).sqrt()
        # Escalado
        weight = self.gamma * weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)

'''class NFBlock(nn.Module):
    """Bloque residual de NFNet"""
    def __init__(self, in_channels, out_channels, stride=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.conv1 = ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ScaledStdConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ScaledStdConv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(x) * self.alpha
        x = self.conv1(x)
        x = F.relu(x) * self.alpha
        x = self.conv2(x)
        return x + residual'''

class ScaledReLU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Convertir a parámetro aprendible

    def forward(self, x):
        return F.relu(x) * self.alpha

class NFBlock(nn.Module):
    """Bloque residual de NFNet"""
    def __init__(self, in_channels, out_channels, stride=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    )
            self.layers = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    ScaledStdConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    ScaledReLU(alpha=self.alpha), 
                    )
        else:
            self.shortcut = nn.Sequential(
                    )
            self.layers = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    ScaledStdConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    )
    def forward(self, x):
        return self.layers(x)

class NFNet(nn.Module):
    """Arquitectura NFNet-F0 simplificada"""
    def __init__(self, num_classes=10, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.neural_network_layers = nn.Sequential(
                #self._make_layer(3, 6, num_blocks=1, stride=1),
                #NFBlock(3, 6, stride=1, alpha=self.alpha),
                #nn.AdaptiveAvgPool2d((1, 1)),
                ScaledStdConv2d(3, 32, kernel_size=3, stride=1, padding=3),
                nn.ReLU(),
                ScaledStdConv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                #ScaledStdConv2d(64, 64, kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                NFBlock(64, 128, stride=1, alpha=self.alpha),
                NFBlock(128, 256, stride=1, alpha=self.alpha),
                NFBlock(256, 512, stride=2, alpha=self.alpha),
                NFBlock(512, 1024, stride=2, alpha=self.alpha),
                #NFBlock(1024, 1024, stride=2, alpha=self.alpha),
                #NFBlock(128, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 256, stride=2, alpha=self.alpha),
                #NFBlock(256, 512, stride=2, alpha=self.alpha),
                #NFBlock(512, 512, stride=2, alpha=self.alpha),
                #NFBlock(512, 512, stride=2, alpha=self.alpha),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        self.dense_neural_network_layers = nn.Sequential(
                #nn.Flatten(start_dim=1),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
                #nn.Linear(512, num_classes)
            )
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [NFBlock(in_channels, out_channels, stride=stride, alpha=self.alpha)]
        for _ in range(1, num_blocks):
            layers.append(NFBlock(out_channels, out_channels, stride=1, alpha=self.alpha))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Inicializar biases a cero

    def forward(self, x):
        x = self.neural_network_layers(x)
        x = self.dense_neural_network_layers(x.squeeze())#x.view(x.size(0), -1))
        return x
