import numpy as np
import random
import matplotlib.pyplot as plt

class OneArmBandit():
    def __init__(self, probs, rewards):
        self.probs = probs
        self.rewards = rewards
        self.expected_value = np.dot(self.rewards, self.probs)

    def act(self):
        return np.random.choice(self.rewards, 1, p=self.probs)[0]


class BernoulliBandit():
    def __init__(self, p):
        self.p = p
        # super().__init__([p, 1-p], [1,0])

    def act(self):
        return random.random() < self.p

    def expected_value(self):
        return self.p


class MultiArmBandit():
    def __init__(self, one_arm_bandits_list):
        self.list_of_bandits = one_arm_bandits_list
        self.expected_returns = [bandit.expected_value() for bandit in self.list_of_bandits]
        self.oracle = np.argmax(self.expected_returns)
        self.max_expected_return = max(self.expected_returns)
        self.rewards_sum  = 0
        self.t = 0

    def act(self, a):
        self.t += 1
        reward = self.list_of_bandits[a].act()
        self.rewards_sum += reward
        return reward

    def regret(self):
        return self.t*self.max_expected_return - self.rewards_sum

    def reset(self):
        self.t = 0
        self.rewards_sum = 0




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
        action_history.append(action)
        regret_history.append(multi_armed_bandit.regret())
    return regret_history

def test_bandit_n_times(multi_armed_bandit, algorithm, T, n_trials):
    regrets = []
    n_bandits = len(multi_armed_bandit.list_of_bandits)
    for t in range(n_trials):
        regrets.append(test_bandit_alg(multi_armed_bandit, algorithm, T))
    min_regret, max_regret, avg_regret = [], [], []
    for t in range(T):
        regrets_at_time_t = [regrets[i][t] for i in range(n_trials)]
        min_regret.append(min(regrets_at_time_t))
        max_regret.append(max(regrets_at_time_t))
        avg_regret.append(sum(regrets_at_time_t)/n_bandits)
    # avg_regret= np.mean(regrets, axis=0)

    final_regrets = [regrets[i][-1] for i in range(n_trials)]
    return min_regret, max_regret, avg_regret, final_regrets

class BanditAlgorithm:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.empirical_rewards = {a : 0 for a in range(n_bandits)}
        self.action_stats = {a: 0 for a in range(n_bandits)}
        self.t = 0

    def update_history(self, last_reward, last_action):
        if last_reward is not None:
            self.empirical_rewards[last_action] += last_reward
        if last_action is not None:
            self.action_stats[last_action] += 1

    def empirical_expected_returns(self):
        def evaluate_bandit(action):
            if self.action_stats[action] > 0:
                return self.empirical_rewards[action] / self.action_stats[action]
            else:
                return float('inf')
        return [evaluate_bandit(a) for a in range(self.n_bandits)]


class Oracle:
    def act(self):
        return 2

class FollowTheLeader(BanditAlgorithm):

    def act(self, last_reward, last_action):
        self.update_history(last_reward, last_action)
        return np.argmax(self.empirical_expected_returns())

class UCB1(BanditAlgorithm):
    def scores(self):
        exploitation_terms = self.empirical_expected_returns()

        t = sum([self.action_stats[x] for x in range(self.n_bandits)])

        def exploration_score(action):
            if self.action_stats[action] > 0:
                return np.sqrt(0.6*np.log(t)/self.action_stats[action])
            else:
                return float('inf')

        exploration_terms = [exploration_score(action) for action in range(self.n_bandits)]
        # print(f'exploit = {exploitation_terms} | explore = {exploration_terms}')
        return np.array(exploitation_terms) + np.array(exploration_terms)

    def act(self, last_reward, last_action):
        self.update_history(last_reward, last_action)
        # print(f'action = {np.argmax(self.scores())}')
        return np.argmax(self.scores())


dupa = MultiArmBandit([
    BernoulliBandit(0.3),
    BernoulliBandit(0.4),
    BernoulliBandit(0.5)])

min_regret, max_regret, avg_regret, final_regrets = test_bandit_n_times(dupa, UCB1(3), 500, 500)

# min_regret, max_regret, avg_regret, final_regrets = test_bandit_n_times(dupa, FollowTheLeader(3), 500, 100)
# regret_history, action_history = test_bandit_alg(dupa, Oracle(), 10000)


plt.clf()
# plt.plot(min_regret, label="Min regret")
# plt.plot(max_regret, label="Max regret")
plt.plot(avg_regret, label="Avg regret")

# plt.hist(final_regrets)
# plt.plot(action_history, label="Regret")

# plt.plot(sarsa_running_avg, label="SARSA")
plt.legend(loc="upper left")
plt.show()

# dupa = MultiArmBandit([
#     OneArmBandit([0.5, 0.5], [0,1]),
#     OneArmBandit([0.2, 0.8], [-1, 1])
# ])

