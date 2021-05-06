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


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
    def select_action(self, state, policy_net):
      rate = self.strategy.get_exploration_rate(self.current_step)
      self.current_step += 1

      if rate > random.random():
          return random.randrange(self.num_actions) # explore      
      else:
          with torch.no_grad():  #to turn off gradient tracking since we're currently using the model for inference and not training. 
              return policy_net(state).argmax(dim=1).to(self.device) # exploit 


