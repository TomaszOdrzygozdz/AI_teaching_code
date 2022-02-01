import random
from copy import copy

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


def make_q_network_sokoban(num_action=4, num_layers=3, kernel_size=(3,3), batch_norm=True, learning_rate=1e-4, weight_decay=0.):
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
    layer = Dense(8, kernel_regularizer=l2(weight_decay), activation='relu')(layer)
    output = Dense(num_action)(layer)

    model = Model(inputs=input_state, outputs=output)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate)
    )
    return model


class DQN_Sokoban:
    def __init__(self, dim_room, num_boxes,
                 epsilon=0.6, min_epsilon=0.1, epsilon_decay=0.999, steps_limit=15,
                 gamma=0.99, q_learning_rate=0.5, replay_buffer_size=10000,
                 replay_batch_size=512, update_target_network=1000, update_after_steps=200):
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.steps_limit = steps_limit
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.update_target_network = update_target_network
        self.update_after_steps = update_after_steps

        self.env = SokobanEnvFast(dim_room=self.dim_room, num_boxes=self.num_boxes)
        self.q_network = make_q_network_sokoban()
        self.target_q_network = make_q_network_sokoban()
        self.replay_buffer = []

        self.all_steps_taken = 0

        self.last_predicted_state = None
        self.last_predicted_action_vals = None

    def predict_q_values(self, state):
        if self.last_predicted_state is not None:
            if np.array_equal(state, self.last_predicted_state):
                action_vals = self.last_predicted_action_vals
            else:
                action_vals = self.q_network.predict(np.array([state]))[0]
        else:
            action_vals = self.q_network.predict(np.array([state]))[0]

        self.last_predicted_state = state
        self.last_predicted_action_vals = action_vals

        return action_vals

    def choose_best_action(self, state):
        action_values = self.predict_q_values(state)
        best_action = np.argmax(action_values)
        return best_action, action_values

    def evaluate_state_batch(self, state_batch):
        action_values = self.target_q_network.predict(np.array(state_batch))
        best_actions = np.argmax(action_values, axis=-1)
        best_vals = np.max(action_values, axis=-1)
        return best_actions, best_vals, action_values

    def choose_action(self, state):

        self.epsilon *= self.epsilon_decay
        best_action, action_values = self.choose_best_action(state)

        if random.random() < self.epsilon + self.min_epsilon:
            return random.randint(0,3), action_values
        else:
            return best_action, action_values

    def choose_action_batch(self, state_batch):
        self.epsilon *= self.epsilon_decay
        best_action, action_values, full_q_values = self.evaluate_state_batch(state_batch)
        chosen_actions = []
        for act in best_action:
            if random.random() < self.epsilon:
                chosen_actions.append(random.randint(0,3))
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
            self.all_steps_taken += 1
            if done:
                save_state(next_state, f'pics/step_{steps}-done', f'step = {steps}-done')
                solved = True
            total_reward += reward
            if steps == self.steps_limit:
                done = True
            new_transition = Transition(state, action_values, action, reward, done, next_state)
            state = next_state
            self.replay_buffer.append(new_transition)

            #update q-function
            if self.all_steps_taken % self.update_after_steps == 0 and self.replay_buffer_size > 4*self.replay_batch_size:
                if len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

                indices = np.random.choice(range(len(self.replay_buffer)), size=self.replay_batch_size)
                transitions = [self.replay_buffer[i] for i in indices]
                train_x, train_y = self.prepare_train_targets(transitions)
                self.q_network.fit(train_x, train_y, epochs=1, verbose=0)

            if self.all_steps_taken % self.update_target_network == 0 and self.all_steps_taken > 0:
                print('updating q  network')
                self.target_q_network.set_weights(self.q_network.get_weights())

        return total_reward, steps, solved

    def run_parallel_episodes(self, env_batch):
        n_envs = len(env_batch)
        done_batch = [False for _ in range(n_envs)]
        reward_batch = [0 for _ in range(n_envs)]
        steps_batch = [0 for _ in range(n_envs)]
        solved_batch = [False for _ in range(n_envs)]
        state_batch = [env.reset() for env in env_batch]
        total_reward = 0

        while not all(done_batch) and min(steps_batch) < self.steps_limit:

            actions, action_values, full_q_values = self.choose_action_batch(state_batch)
            curr_active_envs = 0

            for i in range(n_envs):
                if not done_batch[i]:
                    curr_active_envs += 1
                    action = actions[i]
                    action_val = full_q_values[i]
                    next_state, reward, done, _ = env_batch[i].step(action)

                    self.all_steps_taken += 1

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

                    if self.all_steps_taken % self.update_after_steps == 0 and self.replay_buffer_size > 4 * self.replay_batch_size:
                        if len(self.replay_buffer) > self.replay_buffer_size:
                            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

                        indices = np.random.choice(range(len(self.replay_buffer)), size=self.replay_batch_size)
                        transitions = [self.replay_buffer[i] for i in indices]
                        train_x, train_y = self.prepare_train_targets(transitions)
                        self.q_network.fit(train_x, train_y, epochs=1, verbose=0)

                    if self.all_steps_taken % self.update_target_network == 0 and self.all_steps_taken > 0:
                        print('updating q  network')
                        self.target_q_network.set_weights(self.q_network.get_weights())

        return total_reward / n_envs, sum(solved_batch) / n_envs

    def prepare_train_targets(self, transitions):
        next_state_batch = []
        train_x, train_y = [], []

        for transition in transitions:
            next_state_batch.append(transition.next_state)

        _, next_state_values, _ = self.evaluate_state_batch(next_state_batch)
        for transition, next_state_value in zip(transitions, next_state_values):
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

    # def train(self, epochs):
    #     train_x, train_y =  self.prepare_train_targets()
    #     self.target_q_network.fit(train_x, train_y, epochs=epochs)

    def execute(self, n_episodes, n_envs):
        env_batch = [SokobanEnvFast(dim_room=self.dim_room, num_boxes=self.num_boxes) for _ in range(n_envs)]
        progress = []
        for i in range(n_episodes):
            reward, solved = self.run_parallel_episodes(env_batch)
            print(f'Ep = {i} | Step = {self.all_steps_taken} | reward = {reward} | solved = {solved} | epsilon = {self.epsilon} | buffer = {len(self.replay_buffer)} ')
            if len(progress) > 0:
                progress.append(progress[-1]*0.5 + 0.5*reward)
            else:
                progress.append(reward)
        return progress

    # def full_training(self, n_epochs=10, train_epochs=4):
    #     progress = []
    #     for epoch in range(n_epochs):
    #         reward_avg, solved_avg = self.run_parallel_episodes()
    #         print(f'Step = {epoch*n_epochs} | reward = {reward_avg} | solved = {solved_avg} | epsilon = {self.epsilon} | buffer = {len(self.replay_buffer)} ')
    #         progress.append(reward_avg)
    #         self.train(train_epochs)
    #     self.save_network()
    #     return progress

    def save_network(self):
        self.q_network.save('model_saved')


dupa = DQN_Sokoban((6,6), 1, 0.9)
# env = SokobanEnvFast()

progress = dupa.execute(200, 50)
# rew, sol = dupa.run_parallel_episodes()

# print(f'rew = {rew} sol = {sol}')

plt.clf()
plt.plot(progress, label="DQN learning")
# plt.plot(sarsa_running_avg, label="SARSA")
plt.legend(loc="upper left")
plt.show()

