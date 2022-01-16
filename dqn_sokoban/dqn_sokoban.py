import random

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPool2D, Softmax, Concatenate, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import numpy as np

from sokoban_utils import show_state, save_state

import collections

Transition = collections.namedtuple('transition',
                                    ['state',
                                     'action_values',
                                     'action',
                                     'reward',
                                     'done',
                                     'next_state',
                                     ])

def make_q_network(num_action=4, num_layers=5, kernel_size=(3,3), batch_norm=True, learning_rate=1e-4, weight_decay=0.):
    input_state = Input(batch_shape=(None, None, None, 7))
    layer = input_state

    for _ in range(num_layers):
        layer = Conv2D(
            filters=64,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=l2(weight_decay),
        )(layer)

        if batch_norm:
            layer = BatchNormalization()(layer)

    layer = GlobalAveragePooling2D()(layer)
    # layer = Flatten()(layer)
    layer = Dense(64, kernel_regularizer=l2(weight_decay), activation='relu')(layer)
    output = Dense(num_action)(layer)

    model = Model(inputs=input_state, outputs=output)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    return model


class DQN_Sokoban:
    def __init__(self, dim_room, num_boxes, epsilon=0.9, steps_limit=100, gamma=0.99, q_learning_rate=0.5):
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.epsilon = epsilon
        self.steps_limit = steps_limit
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate

        self.env = SokobanEnvFast(dim_room=self.dim_room, num_boxes=self.num_boxes)
        self.q_network = make_q_network()
        self.replay_buffer = []

    def predict_q_values(self, state):
        return self.q_network.predict(np.array([state]))[0]

    def choose_best_action(self, state):
        action_values = self.predict_q_values(state)
        best_action = np.argmax(action_values)
        return best_action, action_values

    def choose_action(self, state):
        best_action, action_values = self.choose_best_action(state)
        if random.random() < self.epsilon:
            return random.randint(0,3), action_values
        else:
            return best_action, action_values

    def run_one_episode(self):
        done = False
        total_reward = 0
        steps = 0

        state = self.env.reset()
        while not done and steps < self.steps_limit:
            action, action_values = self.choose_action(state)
            save_state(state, f'pics/step_{steps}', f'step = {steps}')
            next_state, reward, done, _ = self.env.step(action)
            steps += 1
            if done:
                save_state(next_state, f'pics/step_{steps}-done', f'step = {steps}-done')
            total_reward += reward
            new_transition = Transition(state, action_values, action, reward, done, next_state)
            state = next_state
            self.replay_buffer.append(new_transition)
        return total_reward, steps

    def collect_experience(self, n_games):
        collected_reward, all_steps = 0, 0
        for num in range(n_games):
            reward, steps = self.run_one_episode()
            collected_reward += reward
            all_steps += steps
            print(f'n = {num} | rew = {collected_reward}')
            print(f'buffer = {len(self.replay_buffer)}')
        return collected_reward / n_games, all_steps / n_games

    def prepare_train_targets(self):
        train_x, train_y = [], []
        for transition in self.replay_buffer:
            state = transition.state
            action = transition.action
            next_state = transition.next_state

            x = state
            y = transition.action_vals

            _, next_state_action_values, = self.choose_best_action(next_state)
            next_state_value = max(next_state_action_values)
            target = transition.reward + self.gamma*next_state_value

            y[action] = (1 - self.q_learning_rate)*y[action] + self.q_learning_rate*target
            train_x.append(x)
            train_y.append(y)

        return train_x, train_y

dupa = DQN_Sokoban((6,6), 1)
env = SokobanEnvFast()

rew, steps = dupa.collect_experience(5)

print(f'r = {rew} steps = {steps}')

# dupa.choose_best_action(np.array([o]))