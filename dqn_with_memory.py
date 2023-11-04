from collections import deque

import gym
import torch

from torch.autograd import Variable
import random
import imageio
import pygame

env = gym.envs.make("MountainCar-v0")
env.reset(seed=1)


class DeepQNetwork:
    def __init__(self, n_state, n_action, n_hidden, lr):
        self.loss = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.loss(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def prepare_experience_to_replay(self, memory, replay_size, gamma):
        if len(memory) < replay_size:
            return

        replay_data = random.sample(memory, replay_size)
        states = []
        td_targets = []
        for state, action, next_state, reward, is_done in replay_data:
            states.append(state)
            q_values = self.predict(state).tolist()
            if is_done:
                q_values[action] = reward
            else:
                q_values_next = self.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()

            td_targets.append(q_values)

        self.update(states, td_targets)


def gen_epsilon_greedy_policy(model, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = model.predict(state)
            return torch.argmax(q_values).item()

    return policy_function


gamma = .9
epsilon = .3
epsilon_decay = .99
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 32
lr = 0.001
n_episode = 1000
total_reward_episode = [0] * n_episode
dqn = DeepQNetwork(n_state, n_action, n_hidden, lr)
memory = deque(maxlen=10000)
size = 20

for episode in range(n_episode):
    policy = gen_epsilon_greedy_policy(dqn, epsilon, n_action)
    state = env.reset()
    is_done = False
    frames = []
    is_render = episode > 950

    steps = 0
    while not is_done:
        steps += 1
        action = policy(state)
        next_state, reward, is_done, _ = env.step(action)
        total_reward_episode[episode] += reward

        if is_render:
            frames.append(env.render(mode='rgb_array'))

        modified_reward = next_state[0] + 0.5

        if next_state[0] >= 0.5:
            modified_reward += 100
        elif next_state[0] >= 0.25:
            modified_reward += 20
        elif next_state[0] >= 0.1:
            modified_reward += 10
        elif next_state[0] >= 0:
            modified_reward += 5

        memory.append((state, action, next_state, modified_reward, is_done))

        if is_done:
            break

        dqn.prepare_experience_to_replay(memory, size, gamma)
        state = next_state

    print('Episode: {}, total reward: {}, epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))

    epsilon = max(epsilon * epsilon_decay, 0.01)

    if is_render:
        imageio.mimsave(f'./{episode}_{steps}.gif', frames, fps=40)

import matplotlib.pyplot as plt

plt.plot(total_reward_episode)
plt.title('Episode reward over time')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
