import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import os
import time
from math import pi
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[-1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return torch.ones((1,)).float().to(DEVICE)

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel

def softmax(xs):
  return F.softmax(xs, dim=1) # B, L

def sigmoid(xs):
  return torch.sigmoid(xs)

def sigmoid_deriv(xs):
  return torch.sigmoid(xs) * (torch.ones_like(xs) - torch.sigmoid(xs))

def arctan_deriv(xs):
    alpha = 10.0
    return 1 / (1 + alpha * xs * xs)


### loss functions
def mse_loss(out, label):
      return torch.sum((out-label)**2)

def mse_deriv(out,label):
      return 2 * (out - label)

def cross_entropy_loss(out,label):
      return nn.CrossEntropyLoss(out,label)

def cross_entropy_deriv(out,label):
      return out - label

### Initialization Functions ###
def gaussian_init(W,mean=0.0, std=0.05):
  return W.normal_(mean=0.0,std=0.05)

def zeros_init(W):
  return torch.zeros_like(W)

def kaiming_init(W, a=math.sqrt(5),*kwargs):
  return init.kaiming_uniform_(W, a)

def xavier_init(W):
  return init.xavier_normal_(W)


def generate_ones_and_minus_ones_matrix(rows, cols):
    random_matrix = torch.randint(0, 2, (rows, cols))
    binary_matrix = torch.where(random_matrix == 0, -1 * torch.ones_like(random_matrix), torch.ones_like(random_matrix))
    return binary_matrix.float()

def set_tensor(xs):
    return xs.float().to(DEVICE)

def cosine_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    # Calculate the cosine annealing factor
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    # Compute the current learning rate based on cosine annealing
    lr = min_lr + (initial_lr - min_lr) * cosine_decay
    return lr

def linear_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    epoch = min(epoch, total_epochs)
    # Calculate the linear decay factor
    linear_decay = 1 - epoch / total_epochs
    # Compute the current learning rate based on linear decay
    lr = min_lr + (initial_lr - min_lr) * linear_decay
    return lr

def none_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    return initial_lr

def BatchNorm2d(x):
    eps = 1e-5
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    return x_normalized

def BatchNorm1d(x, eps=1e-5):

    # Compute mean and variance for the current batch
    batch_mean = x.mean(dim=0)  # Mean over the batch
    batch_var = x.var(dim=0, unbiased=False)  # Variance over the batch
    
    # Normalize the batch
    x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
    
    return x_hat

class SimpleLogger():
    def __init__(self, dirname, filename):
        os.makedirs(dirname, exist_ok=True)
        self.filename = os.path.join(dirname, f"{filename}.log")
        self.file_handler = open(self.filename, 'a')

    def info(self, *args, **kwargs):
        message = " ".join(map(str, args)) + "\n"
        self.file_handler.write(message)
        self.file_handler.flush()

    def __del__(self):
        if not self.file_handler.closed:
            self.file_handler.close()

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))