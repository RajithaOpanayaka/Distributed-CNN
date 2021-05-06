import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T 


class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        #in_features input to the network first layer output=24
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        #output 2 as only 2 actions right or left
        self.out = nn.Linear(in_features=32, out_features=2)
    
    #forward pass
    #input t - any particular image tensor
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t