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
    algorithm.reset()
    regret_history = []
    last_reward = None
    last_action = None

    for num in range(T):
        action = algorithm.act(last_reward, last_action)
        last_action = action
        last_reward = multi_armed_bandit.act(action)
        regret_history.append(multi_armed_bandit.regret())
    return regret_history

def test_bandit_n_times(multi_armed_bandit, algorithm, T, n_trials):
    regrets = []
    for t in range(n_trials):
        regrets.append(test_bandit_alg(multi_armed_bandit, algorithm, T))
    avg_regret= np.mean(regrets, axis=0)
    final_regrets = [regrets[i][-1] for i in range(n_trials)]
    return avg_regret, final_regrets

def test_many_algs_plot(multi_armed_bandit, algs_list, T, n_trials):
    plt.clf()
    for alg in algs_list:
        avg_regret, _ = test_bandit_n_times(multi_armed_bandit, alg, T, n_trials)
        plt.plot(avg_regret, label=alg.name)
    plt.legend(loc="upper left")
    plt.show()


class BanditAlgorithm:
    def __init__(self, n_bandits, name):
        self.n_bandits = n_bandits
        self.name = name

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

    def reset(self):
        self.empirical_rewards = {a: 0 for a in range(self.n_bandits)}
        self.action_stats = {a: 0 for a in range(self.n_bandits)}
        self.t = 0


class Oracle:
    def __init__(self, k):
        self.name = f'Oracle {k}'
        self.k = k

    def act(self, last_reward, last_action):
        return self.k

class FollowTheLeader(BanditAlgorithm):
    def __init__(self, n_bandits):
        super().__init__(n_bandits, 'FTL')

    def act(self, last_reward, last_action):
        self.update_history(last_reward, last_action)
        return np.argmax(self.empirical_expected_returns())

class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, n_bandits, epsilon):
        super().__init__(n_bandits, f'epsilon {epsilon}')
        self.epsilon = epsilon

    def act(self, last_reward, last_action):
        self.update_history(last_reward, last_action)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_bandits-1)
        else:
            return np.argmax(self.empirical_expected_returns())

class UCB1(BanditAlgorithm):
    def __init__(self, n_bandits, alpha):
        super().__init__(n_bandits, f'UCB alpha = {alpha}')
        self.alpha = alpha

    def scores(self):
        exploitation_terms = self.empirical_expected_returns()
        t = sum([self.action_stats[x] for x in range(self.n_bandits)])

        def exploration_score(action):
            if self.action_stats[action] > 0:
                return np.sqrt(self.alpha*np.log(t)/self.action_stats[action])
            else:
                return float('inf')

        exploration_terms = [exploration_score(action) for action in range(self.n_bandits)]
        return np.array(exploitation_terms) + np.array(exploration_terms)

    def act(self, last_reward, last_action):
        self.update_history(last_reward, last_action)
        return np.argmax(self.scores())

class ThompsonSampling():
    name = 'Thompson'
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.reset()

    def act(self, last_reward, last_action):
        if last_reward is not None:
            self.alpha_beta_list[last_action][0] += last_reward
            self.alpha_beta_list[last_action][1] += 1 - last_reward

        thetas = []
        for action in range(self.n_bandits):
            thetas.append(np.random.beta(*self.alpha_beta_list[action]))

        return np.argmax(thetas)

    def reset(self):
        self.alpha_beta_list = [[1, 1] for _ in range(self.n_bandits)]


dupa = MultiArmBandit([
    BernoulliBandit(0.3),
    BernoulliBandit(0.35),
    BernoulliBandit(0.4),
    BernoulliBandit(0.45),
    BernoulliBandit(0.5)])

test_many_algs_plot(dupa, [
    EpsilonGreedy(5, 0.05), UCB1(5, 0.5),FollowTheLeader(5), ThompsonSampling(5)], 10000, 50)

# ttt = test_bandit_alg(dupa, UCB1(3), 500)
# avg_regret_oracle1, final_regrets_oracle1 = test_bandit_n_times(dupa, Oracle(), 100, 1)
# avg_regret_oracle, final_regrets_oracle = test_bandit_n_times(dupa, Oracle(), 100, 10)
# avg_regret_epsilon, final_regrets_epsilon = test_bandit_n_times(dupa, EpsilonGreedy(5, 0.1), 500, 200)
# avg_regret_ucb, final_regrets_ucb = test_bandit_n_times(dupa, UCB1(5, 0.5), 500, 200)
# avg_regret_ftl, final_regrets_ftl = test_bandit_n_times(dupa, FollowTheLeader(5), 500, 200)
# regret_history, action_history = test_bandit_alg(dupa, Oracle(), 10000)


# plt.clf()
# # plt.plot(min_regret, label="Min regret")
# # plt.plot(max_regret, label="Max regret")
# # plt.plot(avg_regret_oracle1, label="Oracle1")
# # plt.plot(avg_regret_oracle, label="Oracle")
# plt.plot(avg_regret_epsilon, label="e-greedy")
# plt.plot(avg_regret_ucb, label="UCB1")
# plt.plot(avg_regret_ftl, label="FTL")
#
# # plt.hist(final_regrets)
# # plt.plot(action_history, label="Regret")
#
# # plt.plot(sarsa_running_avg, label="SARSA")
# plt.legend(loc="upper left")
# plt.show()
#
# # dupa = MultiArmBandit([
#     OneArmBandit([0.5, 0.5], [0,1]),
#     OneArmBandit([0.2, 0.8], [-1, 1])
# ])

