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

        # Full Connections
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.sc2 = nn.Linear(in_features=40, out_features=number_actions)

    def count_neurons(self, image_dim):
        # Create an image
        x = Variable(torch.rand(1, *image_dim))

        # Pass image through convolution layers
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        # Pass image through flattening layers
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        # Propogating the signals
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        # Flatten convolution layer
        x = x.view(x.size(0), -1)

        # Get hidden layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Return output neurons
        return x


# Body
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()

        # Set temperature
        self.T = T

    def forward(self, outputs):
        # Probabilites per action
        probs = F.softmax(outputs * self.T)

        # Final actions to play
        actions = probs.multinomial()


# AI
class AI:
    def __init__(self, brain, body):
        # CNN class
        self.brain = brain

        # SoftmaxBody class
        self.body = body

    def __call__(self, inputs):
        # Receiving input images
        input = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))

        # Propogate images to the brain
        output = self.brain(input)

        # Propogate output to body
        actions = self.body(output)

        # Return actions
        return actions.data.numpy()


# Part 2 - Training the AI with Deep Convolutional Q-Learning
# Get environment
doom_env = image_preprocessing.PreprocessImage(
    SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorrider-v0"))), width=80, height=80, grayscale=True
)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)

# Get actions from environment
number_actions = doom_env.action_space.n

############ AI Instance Start ############
# Brain
cnn = CNN(number_actions)

# Body
softmax_body = SoftmaxBody(T=1.0)

# Instance of AI
ai = AI(brain=cnn, body=softmax_body)
############ AI Instance Finish ############

############ AI Training Start ############
### Experience Replay ###
n_steps = experience_replay.NStepProgress(doom_env, ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)

### Eligibility Trace ###
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        # Get first and last input states
        input = Variable(torch.from_numpy(np.array(series[0].state, series[-1].state, dtype=np.float32)))

        # Get output signal from brain
        output = cnn(input)

        # Get reward
        cumul_reward = 0.0 if series[-1].done else output.data.max()

        # Loop over rewards to get cumulative reward
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward

        #Add first step of series
        state = series[0].state

        #Add first target from output
        target = output[0].data

        #Get the reward for the action permformed
        target[series[0].action] = cumul_reward

        #Update inputs and targets
        inputs.append(state)
        targets.append(target)

    #Return updated input and target
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

### Make the moving average 100 steps ###
class MA:
    #set rewards and size
    def __int__(self, size):
        self.list_of_rewards = []
        self.size = size

    #add cumulative reward
    def __add__(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    #Get average reward
    def average(self):
        return np.mean(self.list_of_rewards)

#Get average of 100 steps
ma = MA(100)

### Train the AI ###
#Mean score error
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
nb_epochs = 100

#loop over num epochs
for epoch in range(1, nb_epochs + 1):

    # Steps per epoch
    memory.run_steps(200)

    # loop over batches of transisitions from epochs
    for batch in memory.sample_batch(128):

        #Get inputs and targets
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)

        #Get predictions and calculate weights
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()

    #Compute cumulative rewards
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()

    #Output results
    print("Epoch: %s, Average Rewards: %s" % (str(epoch), str(avg_reward)))

############ AI Training Finish ############
