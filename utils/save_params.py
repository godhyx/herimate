import copy
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import os
#####################################################
len_param = 0
def save_tensor(tensor,file):
    #return
    global len_param
    tensor = np.array(tensor.detach().data.cpu().squeeze())
    tensor = np.reshape(tensor,-1)
    tensor.tofile(file)
    len_param += len(tensor)
    #print("len :", len_param)

def save_norm_layer(normlayer, file):
    save_tensor(normlayer.weight, file)
    save_tensor(normlayer.bias, file)

def save_linear(linearlayer,file):
    save_tensor(linearlayer.weight,file)
    save_tensor(linearlayer.bias, file)

def save_multiheadatten(multiheadatten,file):
    save_tensor(multiheadatten.in_proj_weight, file)
    save_tensor(multiheadatten.in_proj_bias, file)
    save_tensor(multiheadatten.out_proj.weight, file)
    save_tensor(multiheadatten.out_proj.bias, file) 
##########################################################################
