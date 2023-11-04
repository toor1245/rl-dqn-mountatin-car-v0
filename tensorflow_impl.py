import random
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import imageio

matplotlib.use('TkAgg')

print(gym.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = []

    def append(self, value):
        self.loss.append(self.alpha * self.loss[-1] + (1 - self.alpha) * value if len(self.loss) > 0 else value)

    def get(self):
        return self.loss


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel=''):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec

        self.tic = time.time()

    def plot(self, data, last_episode: bool):
        if not last_episode:
            return

        if time.time() - self.tic > self.sec:
            plt.cla()

            plt.plot(data)

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.show()

            self.tic = time.time()


env = gym.make("MountainCar-v0")
env.reset(seed=1)

n_observations = env.observation_space
print("Environment has observation space =", n_observations)

n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)


### Задаємо агента MountainCar ###

# Задаємо feed-forward нейронну мережу
def create_mountain_car_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])
    return model


### Визначимо фінкцію дій агента ###

# Функція, що бере на вхід множину спостережень, виконує прямий прохід по мережі,
#   і повертає обрану дію.
# Аргументи:
#   model: мережа, що визначає агента
#   observation: спостереження, що даються на вхід моделі
# Вихід:
#   action: вибрана дія агента
def choose_action(model, observation):
    observation = np.expand_dims(observation, axis=0)

    logits = model.predict(observation, verbose=0)
    prob_weights = tf.nn.softmax(logits).numpy()

    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]

    return action


### Функція винагород ###

# Допоміжна функція, що нормалізує масив
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


# Підраховує нормалізовані, скидочні, а також сумарні винагороди (тобто, повернення)
# Аргументи:
#   rewards: винагорода у кожний момент часу
#   gamma: ступінь скидки
# Повертає:
#   Нормалізовану скидочну винагороду
def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


### Крок тренування (forward та backpropagation) ###

def prepare_experience_to_replay(model, memory, replay_size, gamma):
    replay_data = random.sample(memory, replay_size)
    states = []
    td_targets = []
    rewards = []
    for state, action, next_state, reward, is_done in replay_data:
        states.append(state)
        rewards.append(reward)
        q_values = np.array(model.predict(np.expand_dims(state, axis=0), verbose=0)[0])
        if is_done:
            q_values[action] = reward
        else:
            q_values_next = np.array(model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
            q_values[action] = reward + gamma * np.max(q_values_next)

        td_targets.append(q_values)

    return states, td_targets, rewards


def train(model, optimizer, observations, q_values):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = tf.keras.losses.mean_squared_error(y_true=q_values, y_pred=logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def gen_epsilon_greedy_policy(model, epsilon, n_action):
    def policy_function(observation):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            return choose_action(model, observation)
    return policy_function


### Тренування MountainCar ###

# Швидкість навчання та оптимізатор
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

# створення агенту
model = create_mountain_car_model()

# для слідкування за прогресом
smoothed_reward = LossHistory(smoothing_factor=0.9)
plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')
n_episodes = 800
epsilon = .3
epsilon_decay = .99
queue = deque(maxlen=10000)
gamma = 0.95

for i_episode in range(n_episodes):

    is_render = i_episode == n_episodes - 1
    plotter.plot(smoothed_reward.get(), is_render)

    # Перезапустити середовище
    observation = env.reset()
    policy = gen_epsilon_greedy_policy(model, epsilon, n_actions)
    frames = []
    rewards = []

    i = 0
    while True:
        i += 1
        # на основі спостережень, вибрати дію та зробити її в середовищі
        action = policy(observation) # policy(observation) # choose_action(model, observation)
        next_observation, reward, done, info = env.step(action)

        next_x_offset = next_observation[0]

        modified_reward = next_x_offset + 0.5

        if next_x_offset >= 0.5:
            modified_reward += 100
        elif next_x_offset >= 0.25:
            modified_reward += 20
        elif next_x_offset >= 0.1:
            modified_reward += 10
        elif next_x_offset >= 0:
            modified_reward += 5

        if is_render:
            frames.append(env.render(mode='rgb_array'))

        # додати до пам'яті
        queue.append((observation, action, next_observation, modified_reward, done))
        rewards.append(reward)

        # епізод закінчився? сталася аварія, чи все так добре, що ми закінчили?
        if done:
            # Визначити сумарну винагороду та додати її
            total_reward = sum(rewards)
            smoothed_reward.append(total_reward)
            rewards.clear()
            break

        if i >= 20:
            observations, actions, rewards = prepare_experience_to_replay(model, queue, replay_size=20, gamma=gamma)
            train(model, optimizer, np.array(observations), np.array(actions))

        observation = next_observation
        epsilon = max(epsilon * epsilon_decay, 0.01)

    if is_render:
        imageio.mimsave(f'./{i_episode}_{i}.gif', frames, fps=40) 

    print(f"EPISODE: {i_episode}, steps: {i}")
