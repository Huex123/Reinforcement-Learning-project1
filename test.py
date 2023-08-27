import random
import torch
import gym
from gym.wrappers import AtariPreprocessing, LazyFrames, FrameStack
import matplotlib.pyplot as plt

class Qnet(torch.nn.Module):
    '''Q网络'''
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 2)
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(h1)
    
mynet = Qnet()
print(mynet.state_dict())
torch.save(mynet.state_dict(), 'Pong_Qnet.pt')