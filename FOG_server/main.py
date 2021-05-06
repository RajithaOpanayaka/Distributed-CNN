import gym
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

#enviorenment manager
from cartEnv import CartPoleEnvManager

#strategy
from epstrategy import EpsilonGreedyStrategy

#agent
from agent import Agent

#replay memory
from replayMemory import ReplayMemory

#dqn
from dqn import DQN


batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10 #update the target network every 10 episode
memory_size = 100000
lr = 0.001 #learning rate
num_episodes = 1000


#set the device use cpu or gpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#enviorement manager
em = CartPoleEnvManager(device)
#create the strategy
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

#create agent
agent = Agent(strategy, em.num_actions_available(), device)
#create replay memory
memory = ReplayMemory(memory_size)

#create policy network and target network
#pass height and width to create appropriate input shape

policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)