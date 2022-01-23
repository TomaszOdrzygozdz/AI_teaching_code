import numpy as np
import random
import matplotlib.pyplot as plt

class OneArmBandit():
    def __init__(self, probs, rewards):
        self.probs = probs
        self.rewards = rewards

    def act(self):
        return np.random.choice(self.rewards, 1, p=self.probs)[0]

    def expected_value(self):
        return np.dot(self.rewards, self.probs)

class BernoulliBandit(OneArmBandit):
    def __init__(self, p):
        super().__init__([p, 1-p], [1,0])



class MultiArmBandit():
    def __init__(self, one_arm_bandits_list):
        self.list_of_bandits = one_arm_bandits_list
        self.history_rewards = []
        self.history_actions = []
        self.expected_returns = [bandit.expected_value() for bandit in self.list_of_bandits]
        self.oracle = np.argmax(self.expected_returns)
        self.max_expected_return = max(self.expected_returns)

    def act(self, a):
        reward = self.list_of_bandits[a].act()
        self.history_rewards.append(reward)
        return self.list_of_bandits[a].act()

    def regret(self):
        T = len(self.history_rewards)
        # print(f'history = {self.history_rewards}')
        return T*self.max_expected_return - sum(self.history_rewards)

    def reset(self):
        self.history_rewards = []
        self.history_actions = []




def test_bandit_alg(multi_armed_bandit, algorithm, T):
    multi_armed_bandit.reset()
    regret_history = []
    action_history = []
    last_reward = None
    last_action = None

    for num in range(T):
        action = algorithm.act(last_reward, last_action)
        last_action = action
        last_reward = multi_armed_bandit.act(action)
        # print(f'action = {action}')
        action_history.append(action)
        regret_history.append(multi_armed_bandit.regret())
    return regret_history

def test_bandit_n_times(multi_armed_bandit, algorithm, T, n_trials):
    regrets = {i: test_bandit_alg(multi_armed_bandit, algorithm, T) for i in range(n_trials)}
    min_regret, max_regret, avg_regret = [], [], []
    for t in range(T):
        regrets_at_time_t = [regrets[i][t] for i in range(n_trials)]
        min_regret.append(min(regrets_at_time_t))
        max_regret.append(max(regrets_at_time_t))
        avg_regret.append(np.mean(regrets_at_time_t))

    final_regrets = [regrets[i][-1] for i in range(n_trials)]
    return min_regret, max_regret, avg_regret, final_regrets

class Oracle:
    def act(self):
        return 2

class FollowTheLeader:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.empirical_rewards = {a : [] for a in range(n_bandits)}

    def act(self, last_reward, last_action):
        if last_reward is not None:
            self.empirical_rewards[last_action].append(last_reward)
        def evaluate_bandit(list_of_rewards):
            if len(list_of_rewards) > 0:
                return np.mean(list_of_rewards)
            else:
                return float('inf')
        emprical_expected_returns = [evaluate_bandit(self.empirical_rewards[a]) for a in range(self.n_bandits)]
        # print(f'empirical_expected_retruns = {emprical_expected_returns}')

        return np.argmax(emprical_expected_returns)



dupa = MultiArmBandit([
    BernoulliBandit(0.3),
    BernoulliBandit(0.4),
    BernoulliBandit(0.5)])

min_regret, max_regret, avg_regret, final_regrets = test_bandit_n_times(dupa, FollowTheLeader(3), 500, 100)
# regret_history, action_history = test_bandit_alg(dupa, Oracle(), 10000)

plt.clf()
plt.plot(min_regret, label="Min regret")
plt.plot(max_regret, label="Max regret")
plt.plot(avg_regret, label="Avg regret")

# plt.plot(action_history, label="Regret")

# plt.plot(sarsa_running_avg, label="SARSA")
plt.legend(loc="upper left")
plt.show()

# dupa = MultiArmBandit([
#     OneArmBandit([0.5, 0.5], [0,1]),
#     OneArmBandit([0.2, 0.8], [-1, 1])
# ])

