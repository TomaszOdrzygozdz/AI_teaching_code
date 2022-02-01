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
                                     'action',
                                     'reward',
                                     'done',
                                     'next_state',
                                     ])

def make_cartpole_network(input_size=4, num_action=2, num_layers=3, learning_rate=1e-4, weight_decay=0.):
    input_state = Input(batch_shape=(None, input_size))
    layer = input_state

    for _ in range(num_layers):
        layer = Dense(64,
            activation='relu',
            kernel_regularizer=l2(weight_decay),
        )(layer)

    output = Dense(num_action)(layer)
    model = Model(inputs=input_state, outputs=output)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    return model

class ModifiedCartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        if done:
            reward = -10
        return obs, reward/10, done, {}


class DQN_Cartpole:
    def __init__(self, epsilon=1.2,
                 min_epsilon=0.1,
                 epsilon_decay=0.9995,
                 gamma=0.99,
                 learning_rate = 0.5,
                 replay_buffer_size=2000,
                 train_every_n_steps=32,
                 train_mini_batch_size=128,
                 update_target_network_every_n_steps=128,
                 train_epochs=2):

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.train_every_n_steps = train_every_n_steps
        self.train_mini_batch_size = train_mini_batch_size
        self.update_target_network_every_n_steps = update_target_network_every_n_steps
        self.train_epochs = train_epochs

        # self.env = gym.make('CartPole-v0')
        self.env = ModifiedCartPole()
        self.q_network = make_cartpole_network()
        self.target_network = make_cartpole_network()
        self.replay_buffer = []
        self.total_steps_taken = 0

        self.do_decay = True

        self.global_stats = [1,1]

    def predict_q_values(self, state):
        predi = self.q_network.predict(np.array([state]))[0]
        if random.random() < 0.1:
            predi2 = self.target_network.predict(np.array([state]))[0]
            print(f'target predi = {predi2}')
        return predi

    def choose_best_action(self, state):
        action_values = self.predict_q_values(state)
        best_action = np.argmax(action_values)
        self.global_stats[best_action] += 1
        return best_action

    def evaluate_state_batch(self, state_batch):
        action_values = self.target_network.predict(np.array(state_batch))
        best_actions = np.argmax(action_values, axis=-1)
        best_vals = np.max(action_values, axis=-1)
        return best_actions, best_vals, action_values

    def choose_action(self, state):
        if self.epsilon > self.min_epsilon and self.do_decay:
            self.epsilon *= self.epsilon_decay
        if random.random() < self.epsilon:
            return random.randint(0,1)
        else:
            return self.choose_best_action(state)

    def run_one_episode(self):
        done = False
        total_reward = 0
        steps_taken = 0
        state = self.env.reset()
        ep_actions = []
        while not done:
            action = self.choose_action(state)
            ep_actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            steps_taken += 1
            self.total_steps_taken += 1
            total_reward += reward
            new_transition = Transition(state, action, reward, done, next_state)
            state = next_state
            self.replay_buffer.append(new_transition)

            if self.total_steps_taken % self.train_every_n_steps == 0 and len(self.replay_buffer) > self.train_mini_batch_size:
                self.train(self.train_epochs)
                if self.total_steps_taken % self.update_target_network_every_n_steps == 0:
                    print(f'updating')
                    self.target_network.set_weights(self.q_network.get_weights())

        # print(f'right = {sum(ep_actions)/len(ep_actions)} | epizode actions = {ep_actions}')
        return steps_taken

    def prepare_train_targets(self):

        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

        if len(self.replay_buffer) > self.train_mini_batch_size:
            indices = random.sample(range(len(self.replay_buffer)), self.train_mini_batch_size)
        else:
            indices = list(range(len(self.replay_buffer)))

        replay_batch = [self.replay_buffer[i] for i in indices]

        next_state_batch = []
        state_batch = []
        train_x, train_y = [], []

        for transition in replay_batch:
            next_state_batch.append(transition.next_state)
            state_batch.append(transition.state)

        _, next_state_values, _ = self.evaluate_state_batch(next_state_batch)
        _, _, state_action_vals = self.evaluate_state_batch(state_batch)



        for transition, state_vals, next_state_value in zip(replay_batch, state_action_vals, next_state_values):
            action = transition.action
            x = transition.state.copy()
            y = state_vals.copy()
            if not transition.done:
                target = transition.reward + self.gamma*next_state_value
            else:
                target = transition.reward
            y[action] = (1-self.learning_rate)*y[action] + self.learning_rate * target
            train_x.append(x)
            train_y.append(y)

        # for transition in terminal_states:
        #     print('use terminal')
        #     x = copy(transition.state)
        #     y = np.array([0,0])
        #     train_x.append(x)
        #     train_y.append(y)

        return np.array(train_x), np.array(train_y)

    def train(self, epochs):
        train_x, train_y =  self.prepare_train_targets()
        self.q_network.fit(train_x, train_y, epochs=epochs, verbose=0)

    def full_training(self, n_epochs):
        progress = []
        progress_moving_avg = [0]
        for epoch in range(n_epochs):
            reward= self.run_one_episode()
            progress_moving_avg.append(progress_moving_avg[-1]*0.75 + 0.25*reward)
            stats = [self.global_stats[0]/sum(self.global_stats), self.global_stats[1]/sum(self.global_stats)]
            print(f'Episode = {epoch} | reward = {reward} | epsilon = {self.epsilon} | buffer = {len(self.replay_buffer)} | global_stats = {stats}')
            progress.append(reward)

        # self.save_network()
        return progress, progress_moving_avg

    def save_network(self):
        self.q_network.save('model_saved')


dupa = DQN_Cartpole()

progress, smooth_progress = dupa.full_training(300)
# rew, sol = dupa.run_parallel_episodes()

# print(f'rew = {rew} sol = {sol}')

plt.clf()
plt.plot(progress, label="DQN learning")
plt.plot(smooth_progress, label="DQN learning (smooth)")
plt.legend(loc="upper left")
plt.show()

