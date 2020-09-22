# AI for Doom


# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing


# Part 1 - Building the AI
# Brain
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()

        # Convolution Layers
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)

        #Full Connections
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.sc2 = nn.Linear(in_features=40, out_features=number_actions)

    def count_neurons(self, image_dim):
        #Create an image
        x = Variable(torch.rand(1, *image_dim))

        # Pass image through convolution layers
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        #Pass image through flattening layers
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        #Propogating the signals
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        #Flatten convolution layer
        x = x.view(x.size(0), -1)

        #Get hidden layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #Return output neurons
        return x
# Body
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()

        #Set temperature
        self.T = T

    def forward(self, outputs):
        #Probabilites per action
        probs = F.softmax(outputs * self.T)

        #Final actions to play
        actions = probs.multinomial()
#AI
class AI:
    def __init__(self, brain, body):
        #CNN class
        self.brain = brain

        #SoftmaxBody class
        self.body = body

    def __call__(self, inputs):
        #Receiving input images
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))

        #Propogate images to the brain
        output = self.brain(input)

        #Propogate output to body
        actions = self.body(output)

        #Return actions
        return actions.data.numpy()
# Part 2 - Training the AI with Deep Convolutional Q-Learning
