import gym
from gym import spaces
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class GridWorldEnv(gym.Env):
    def __init__(self):
        # Define the state space and action space
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([7, 7]), dtype=np.int)
        self.action_space = spaces.Discrete(4)

        # Define the initial state and terminal states
        self.start_state = np.array([1, 1])
        self.goal_states = [np.array([5, 6])]
        
        self.blocked_states = [np.array([4,1]), np.array([4,2]), np.array([4,3]), np.array([4,4]), np.array([1,4]), np.array([2,4]), np.array([3,4])]

        # Define the reward structure
        self.reward_dict = {
            tuple(self.goal_states[0]): 1,
        }
        for state in self.blocked_states:
            self.reward_dict[tuple(state)] = -1

        # Define the transition dynamics
        self.transition_dict = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }

        self.current_state = None
        self.steps = 0

    def reset(self):
        self.current_state = self.start_state
        self.steps = 0
        return self.current_state

    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid action")

        next_state = self.current_state + self.transition_dict[action]
        next_state = np.clip(next_state, self.observation_space.low, self.observation_space.high)
        reward = self.reward_dict.get(tuple(next_state), 0)
        # if next_state is in the blocked states, then the agent stays in the same state
        if tuple(next_state) in self.reward_dict and self.reward_dict[tuple(next_state)] == -1:
            next_state = self.current_state
        done = tuple(next_state) in self.reward_dict and self.reward_dict[tuple(next_state)] == 1
        self.current_state = next_state
        self.steps += 1
        return next_state, reward, done, {}
    
    def get_world_size(self):
        return self.observation_space.high + 1

    def render(self, mode='human'):
        if mode == 'human':
            # Create a grid of zeros representing the world
            world = np.zeros(self.get_world_size(), dtype=str)

            # Set the blocked and goal states
            for state in self.blocked_states:
                world[tuple(state)] = 'X'
            for state in self.goal_states:
                world[tuple(state)] = 'G'

            # Set the current state of the agent
            world[tuple(self.current_state)] = 'A'

            # Print the grid
            for row in world:
                print(' '.join(row))
                print()
        else:
            super().render(mode=mode)
    

env = GridWorldEnv()

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.to(dtype=x.dtype)

class Agent(object):
    def __init__(self, input_shape, num_actions, gamma=0.99, epsilon=1.0, lr=0.001, batch_size=64, capacity=10000):
        self.state_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.memory = ReplayMemory(capacity)
        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transition = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transition))
        
        non_goal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_goal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_goal_mask] = self.target_net(non_goal_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        self.loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        

env = GridWorldEnv()
agent = Agent(env.observation_space.shape[0], env.action_space.n)

# Training
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

num_episodes = 100

for episode in range(num_episodes):
    state = torch.FloatTensor(env.reset()).to(device)
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        # next_state = torch.FloatTensor(next_state).to(device)
        agent.memory.push(torch.FloatTensor(state).to(device), action, torch.FloatTensor(next_state).to(device), torch.FloatTensor([reward]).to(device))
        state = next_state
        agent.optimize_model()
    episode_durations.append(env.steps + 1)
    plot_durations()

    if episode % 10 == 0:
        print("Episode: {}, Steps: {}".format(episode, env.steps + 1))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
env.render()

# Testing
state = torch.Tensor(env.reset()).to(device)
done = False
while not done:
    env.render()
    action = agent.policy_net(state).max(1)[1].view(1, 1)
    next_state, reward, done, _ = env.step(action.item())
    state = torch.Tensor(next_state).to(device)
env.close()
