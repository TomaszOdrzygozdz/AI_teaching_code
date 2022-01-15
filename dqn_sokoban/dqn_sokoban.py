import random

from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPool2D, Softmax, Concatenate, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import numpy as np

from sokoban_utils import show_state

import collections

Transition = collections.namedtuple('transition',
                                    ['state',
                                     'action_values',
                                     'action',
                                     'reward',
                                     'done'
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
    def __init__(self, dim_room, epsilon=0.2):
        self.dim_room = dim_room
        self.epsilon = epsilon

        self.env = SokobanEnvFast(dim_room=self.dim_room)
        self.q_network = make_q_network()
        self.replay_buffer = []

    def predict_q_values(self, state):
        return self.q_network.predict(np.array([state]))[0]

    def choose_best_action(self, state):
        action_values = self.predict_q_values(state)
        best_action = np.argmax(action_values)
        return best_action, action_values

    def choose_action(self, state):
        if random.random() < self.epsilon

dupa = DQN_Sokoban((3,3))
env = SokobanEnvFast()
o = env.reset()
dupa.choose_best_action(np.array([o]))