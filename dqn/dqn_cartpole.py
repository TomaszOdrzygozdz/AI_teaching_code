import random
from copy import copy

import gym
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import numpy as np

from sokoban_utils import show_state, save_state

import matplotlib.pyplot as plt
import collections

Transition = collections.namedtuple('transition',
                                    ['state',
                                     'action_values',
                                     'action',
                                     'reward',
                                     'done',
                                     'next_state',
                                     ])

def make_cartpole_network(input_size=4, num_action=2, num_layers=3, batch_norm=True, learning_rate=1e-4, weight_decay=0.):
    input_state = Input(batch_shape=(None, input_size))
    layer = input_state

    for _ in range(num_layers):
        layer = Dense(256,
            activation='relu',
            kernel_regularizer=l2(weight_decay),
        )(layer)

        if batch_norm:
            layer = BatchNormalization()(layer)

    output = Dense(num_action)(layer)

    model = Model(inputs=input_state, outputs=output)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    return model


class DQN_Cartpole:
    def __init__(self, n_envs, epsilon=0.6, min_epsilon=0.1, epsilon_decay=0.999, steps_limit=30, gamma=0.99, q_learning_rate=0.5, replay_buffer_size=10000):

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.steps_limit = steps_limit
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate
        self.replay_buffer_size = replay_buffer_size

        self.env_batch = [gym.make('CartPole-v0') for _ in range(n_envs)]

        self.env = gym.make('CartPole-v0')
        self.q_network = make_cartpole_network()
        self.replay_buffer = []

    def predict_q_values(self, state):
        return self.q_network.predict(np.array([state]))[0]

    def choose_best_action(self, state):
        action_values = self.predict_q_values(state)
        best_action = np.argmax(action_values)
        return best_action, action_values

    def evaluate_state_batch(self, state_batch):
        action_values = self.q_network.predict(np.array(state_batch))
        best_actions = np.argmax(action_values, axis=-1)
        best_vals = np.max(action_values, axis=-1)
        return best_actions, best_vals, action_values

    def choose_action(self, state):
        self.epsilon *= self.epsilon_decay
        best_action, action_values = self.choose_best_action(state)
        if random.random() < self.epsilon + self.min_epsilon:
            return random.randint(0,1), action_values
        else:
            return best_action, action_values

    def choose_action_batch(self, state_batch):
        self.epsilon *= self.epsilon_decay
        best_action, action_values, full_q_values = self.evaluate_state_batch(state_batch)
        chosen_actions = []
        for act in best_action:
            if random.random() < self.epsilon:
                chosen_actions.append(random.randint(0,1))
            else:
                chosen_actions.append(act)
        return chosen_actions, action_values, full_q_values

    def run_one_episode(self):
        done = False
        total_reward = 0
        steps = 0
        solved = False
        state = self.env.reset()
        while not done and steps < self.steps_limit:
            action, action_values = self.choose_action(state)
            save_state(state, f'pics/step_{steps}', f'step = {steps}')
            next_state, reward, done, _ = self.env.step(action)
            steps += 1
            if done:
                save_state(next_state, f'pics/step_{steps}-done', f'step = {steps}-done')
                solved = True
            total_reward += reward
            if steps == self.steps_limit:
                done = True
            new_transition = Transition(state, action_values, action, reward, done, next_state)
            state = next_state
            self.replay_buffer.append(new_transition)
        return total_reward, steps, solved

    def run_parallel_episodes(self):
        n_envs = len(self.env_batch)
        done_batch = [False for _ in range(n_envs)]
        reward_batch = [0 for _ in range(n_envs)]
        steps_batch = [0 for _ in range(n_envs)]
        solved_batch = [False for _ in range(n_envs)]
        state_batch = [env.reset() for env in self.env_batch]
        total_reward = 0

        while not all(done_batch) and min(steps_batch) < self.steps_limit:

            actions, action_values, full_q_values = self.choose_action_batch(state_batch)
            curr_active_envs = 0

            for i in range(n_envs):
                if not done_batch[i]:
                    curr_active_envs += 1
                    action = actions[i]
                    action_val = full_q_values[i]
                    next_state, reward, done, _ = self.env_batch[i].step(action)
                    if done:
                        solved_batch[i] = True
                    reward_batch[i] = reward

                    steps_batch[i] += 1
                    if steps_batch[i] > self.steps_limit:
                        done = True
                    done_batch[i] = done

                    total_reward += reward
                    new_transition = Transition(state_batch[i], action_val, action, reward, done, next_state)
                    self.replay_buffer.append(new_transition)

                    state_batch[i] = copy(next_state)
        return total_reward / n_envs, sum(solved_batch) / n_envs

    def prepare_train_targets(self):

        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

        next_state_batch = []
        train_x, train_y = [], []

        for transition in self.replay_buffer:
            next_state_batch.append(transition.next_state)

        _, next_state_values, _ = self.evaluate_state_batch(next_state_batch)

        for transition, next_state_value in zip(self.replay_buffer, next_state_values):
            action = transition.action

            x = copy(transition.state)
            y = copy(transition.action_values)

            if not transition.done:
                target = transition.reward + self.gamma*next_state_value
            else:
                target = transition.reward
            smoothed_target = (1 - self.q_learning_rate)*y[action] + self.q_learning_rate*target
            y[action] = smoothed_target
            train_x.append(x)
            train_y.append(y)

        return np.array(train_x), np.array(train_y)

    def train(self, epochs):
        train_x, train_y =  self.prepare_train_targets()
        self.q_network.fit(train_x, train_y, epochs=epochs)

    def full_training(self, n_epochs=10, train_epochs=4):
        progress = []
        for epoch in range(n_epochs):
            reward_avg, solved_avg = self.run_parallel_episodes()
            print(f'Step = {epoch*n_epochs} | reward = {reward_avg} | solved = {solved_avg} | epsilon = {self.epsilon} | buffer = {len(self.replay_buffer)} ')
            progress.append(reward_avg)
            self.train(train_epochs)
        self.save_network()
        return progress

    def save_network(self):
        self.q_network.save('model_saved')


dupa = DQN_Cartpole(5)


progress = dupa.full_training(200, 2)
# rew, sol = dupa.run_parallel_episodes()

# print(f'rew = {rew} sol = {sol}')

plt.clf()
plt.plot(progress, label="DQN learning")
plt.legend(loc="upper left")
plt.show()

