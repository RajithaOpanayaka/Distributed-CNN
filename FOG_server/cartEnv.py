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



class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
    def reset(self):
      self.env.reset()
      self.current_screen = None
    def close(self):
      self.env.close()
    def render(self, mode='human'):
      return self.env.render(mode)
    def num_actions_available(self):
      return self.env.action_space.n
    def take_action(self, action):  
      #reward and wethere the episode is ended      
      _, reward, self.done, _ = self.env.step(action.item()) # item() just returns the value of this tensor as a standard Python number
      return torch.tensor([reward], device=self.device)#returns the reward wrapped in this PyTorch tensor
    def just_starting(self):
      return self.current_screen is None
    def just_starting(self):
      return self.current_screen is None
    def get_state(self):
      #
      if self.just_starting() or self.done:
          self.current_screen = self.get_processed_screen()
          black_screen = torch.zeros_like(self.current_screen)
          return black_screen
      else:
          s1 = self.current_screen
          s2 = self.get_processed_screen()
          self.current_screen = s2
          return s2 - s1
    def get_screen_height(self):
      screen = self.get_processed_screen()
      return screen.shape[2]

    def get_screen_width(self):
      screen = self.get_processed_screen()
      return screen.shape[3]
    def get_processed_screen(self):
      screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
      screen = self.crop_screen(screen)
      return self.transform_screen_data(screen)
    def crop_screen(self, screen):
      screen_height = screen.shape[1]

      # Strip off top and bottom
      top = int(screen_height * 0.4)
      bottom = int(screen_height * 0.8)
      screen = screen[:, top:bottom, :]
      return screen
    def transform_screen_data(self, screen):       
      # Convert to float, rescale, convert to tensor
      screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
      screen = torch.from_numpy(screen)

      # Use torchvision package to compose image transforms
      resize = T.Compose([
          T.ToPILImage()
          ,T.Resize((40,90))
          ,T.ToTensor()
      ])

      return resize(screen).unsqueeze(0).to(self.device) # add a batch dimension (BCHW)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)

screen = em.get_processed_screen()

plt.figure()
plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
plt.title('Processed screen example')
plt.show()