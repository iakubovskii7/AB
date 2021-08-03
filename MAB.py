import re
from collections import Counter
from random import choices
from typing import List, Dict
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
import enum
import sys, os

import numpy as np
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from bootstrapped_ways import bootstrap_jit_parallel

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def gener_random_revenue(revenue_values: List):
    """
    :param revenue_values: historical revenue values
    :return: random value according to historical distribution
    """
    revenue_counter = Counter(revenue_values)
    values = list(revenue_counter.keys())
    probs = list(map(lambda x: x / len(revenue_values), list(revenue_counter.values())))
    return choices(values, probs)

# utility_dict - накопленные суммы по каждой ручки (reward)
# selections_dist - кажыдй ключ - ручки, значение сколько раз выбрали руку
# explore_coefficient = 2 (есть разные эвристики); важный для expl-expl - исследуешь и узнаешь; с 2 все круто и можно уменьшать
# могу уменьшать прямо во время эксперимента (с каждым шагом уменьшаем на 1%) с каждым шагом должен уменьшаться на log() -


def ucb(utility_dict: Dict[str, float],
        selections_dict: Dict[str, int],
        explore_coefficient: float = 2.0
        ) -> str:
    """
    Upper Confidence Bounds
    :param utility_dict: keys - handles, values - performance score;
    :param selections_dict: keys - handles, values - count of selections some handle;
    :param explore_coefficient: heuristic value for bound;

    :return: action;
    """
    rewards = np.fromiter(utility_dict.values(), dtype=float) / (np.fromiter(selections_dict.values(), dtype=float)
                                                                 + 1e-05)
    selections = np.fromiter(selections_dict.values(), dtype=float)
    if sum(selections) != 0:
        sel_sum_log = np.log(sum(selections))
    else:
        sel_sum_log = 0
    n_action = int(np.argmax(
        rewards + explore_coefficient * np.sqrt(sel_sum_log / (selections + 1e-5))
    ))
    return [*utility_dict.keys()][n_action]


def get_bootstrap_upper_bound(revenues: np.array) -> float:
    """
    :return: 95% upper bounds
    """
    revenues = revenues + 1e-05
    bootstr_mean = bs.bootstrap(stat_func=bs_stats.mean,
                                values=revenues,
                                num_threads=1,
                                num_iterations=1e04)
    conf_int = re.findall(r"\(.*", str(bootstr_mean))[0]
    upper_conf = float(conf_int.strip(")").split(", ")[1])
    return(upper_conf)


def ucb_bootstrapped(revenue_dict: Dict[str, np.array],
                     n_boots: int
                     ) -> str:
    """
    Upper Confidence Bounds with Bootstapped Upper Bounds
    :param revenue_dict: keys - handles, values - historical numpy array of revenues

    :return: action;
    """

    # Find upper bounds with bootstrapped samples
    upper_bounds = list(map(bootstrap_jit_parallel, revenue_dict.values(),
                            [n_boots]*len(revenue_dict.keys())))

    n_action = int(np.argmax(upper_bounds))
    return [*revenue_dict.keys()][n_action]


class BernoulliBandit:
    def __init__(self, arms=2):
        self._probs = np.random.random(arms)

    @property
    def action_count(self):
        return len(self._probs)

    def pull(self, action):
        if np.any(np.random.random() > self._probs[action]):
            return 0.0
        else:
            return 1.0

    def optimal_reward(self):
        """
        Used for regret calculation
        """
        return np.max(self._probs)

    def step(self):
        """
        used in non-stationary version
        """
        pass

    def reset(self):
        """Used in non-stationary version
        """


class AbstractAgent(metaclass=ABCMeta):
    def init_actions(self, arms):
        self._successes = np.zeros(arms)
        self._failures = np.zeros(arms)
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Get current best action
        :rtype: int
        """
        pass
    def update(self, action, reward):
        """
        Observe reward from action and update agent's internal parameters
        :type action: int
        :type reward: int
        """
        self._total_pulls += 1
        amount_of_success = int(np.random.random() * 100)
        if reward == 1:
            self._successes[action] += amount_of_success
        else:
            self._failures[action] += 100 - amount_of_success

    @property
    def name(self):
        return self.__class__.__name__


class RandomAgent(AbstractAgent):
    def get_action(self):
        return np.random.randint(0, len(self._successes))


class EpsilonGreedyAgent(AbstractAgent):
    def __init__(self, epsilon=0.01):
        self._epsilon = epsilon

    def get_action(self):
        p = self._successes / (self._successes + self._failures + 1e-10)

        if np.random.random() > self._epsilon:
            return np.argmax(p)
        else:
            return np.random.randint(0, len(self._successes))

    @property
    def name(self):
        return self.__class__.__name__ + "(epsilon = {})".format(self._epsilon)


class UCBAgent(AbstractAgent):
    def get_action(self):
        ucb_result = np.sqrt(2 * np.log1p(self._total_pulls) / (self._successes + self._failures + 1e-10))
        w = self._successes / (self._successes + self._failures + 1e-10) + ucb_result
        return np.argmax(w)

    @property
    def name(self):
        return self.__class__.__name__


class ThompsonSamplingAgent(AbstractAgent):
    def get_action(self):
        """
        :eps = 1e-12
        :weights: np.zeros_like(self._successes) - not TS
        :weights: np.random.beta(self._successes + eps, self._failures + eps)
        :return: arm
        """
        theta = np.random.beta(self._successes+1, self._failures+1)
        return np.argmax(theta)
    @property
    def name(self):
        return self.__class__.__name__


class BernoulliBanditBatch:
    def __init__(self, experiment_df, n_chunks=100, arms=2):
        # self._probs = np.random.random(arms)
        self._experiment_df = experiment_df
        self._n_chunks = n_chunks
        self._n_actions = arms

    @property
    def action_count(self):
        return len(self._experiment_df.keys())

    def pull(self, action, iter):
        return self._experiment_df[action][iter*50 : (iter+1)*50]

    def optimal_reward(self, iter):
        """
        Used for regret calculation
        """
        all_rewards_iter = np.array([np.sum(self._experiment_df[i][iter * self._n_chunks : (iter+1) * self._n_chunks])
                                    for i, key in enumerate(self._experiment_df.keys())
                                    ])
        return np.max(all_rewards_iter)

    def step(self):
        """
        used in non-stationary version
        """
        pass

    def reset(self):
        """Used in non-stationary version
        """


class AbstractAgentBatch(metaclass=ABCMeta):
    """
    Get action based on input data
    """
    def init_actions(self, arms):
        self._successes = np.zeros(arms)
        self._failures = np.zeros(arms)
        self._total_pulls = 0

    @abstractmethod
    def get_action(self):
        """
        Get current best action
        :rtype: int
        """
        pass

    def update(self, action, reward):
        """
        Observe reward from action and update agent's internal parameters
        :type action: int
        :type reward: list of rewards !
        """
        self._total_pulls += len(reward)
        amount_of_success = np.count_nonzero(reward)
        self._successes[action] += amount_of_success
        self._failures[action] += len(reward) - np.count_nonzero(reward)

    @property
    def name(self):
        return self.__class__.__name__


class ThompsonSamplingAgentBatch(AbstractAgentBatch):
    def get_action(self):
        """
        :eps = 1e-12
        :weights: np.zeros_like(self._successes) - not TS
        :weights: np.random.beta(self._successes + eps, self._failures + eps)
        :return: arm
        """
        theta = np.random.beta(self._successes+1, self._failures+1)
        return np.argmax(theta)
    @property
    def name(self):
        return self.__class__.__name__


def get_results(env, agents, n_steps=5000, n_trials=50):
    regret_scores = OrderedDict({
        agent.name: [0.0 for step in range(n_steps)] for agent in agents
    })
    reward_scores = OrderedDict({
        agent.name: [tuple(0 for i in range(env.arms)) for step in range(n_steps)] for agent in agents
    })
    # prob_win_scores = OrderedDict({
    #     agent.name: [0.0 for step in range(n_steps)] for agent in agents
    # })

    for trial in range(n_trials):
        env.reset()

        for a in agents:
            a.init_actions(env.action_count)
        for i in range(n_steps):
            try:
                optimal_reward = env.optimal_reward(i)
            except:  # generate random in every iter
                optimal_reward = env.optimal_reward()

            for agent in agents:
                action = agent.get_action()
                try:
                    reward = env.pull(action, i)  
                except:  # generate random in every iter
                    reward = env.pull(action)
                agent.update(action, reward)
                regret_scores[agent.name][i] += optimal_reward - np.sum(reward)
                reward_scores[agent.name][i][action] = reward

            env.step()  # change bandit step if it is unstationary

    for agent in agents:
        regret_scores[agent.name] = np.cumsum(regret_scores[agent.name]) / n_trials
        for arm in env.arms:
            reward_scores[agent.name][arm] = np.cumsum(reward_scores[agent.name][arm])

    return regret_scores, reward_scores


def plot_regret(agents, regret_scores, plot_name):
    for agent in agents:
        plt.plot(regret_scores[agent.name])


    plt.legend([agent.name for agent in agents])

    plt.ylabel("regret")
    plt.xlabel("steps")

    plt.savefig("Data/Plots/Regret_" + plot_name + ".pdf")


def plot_reward(agents, reward_scores, plot_name):
    for agent in agents:
        for arm, _ in enumerate(reward_scores[agent.name]):
            plt.plot(reward_scores[agent.name][arm])
    plt.legend([(agent.name, arm) for agent, arms in zip(agents, len(reward_scores[agent.name]))])
    plt.ylabel("rewards")
    plt.xlabel("steps")
    plt.savefig("Data/Plots/Rewards_" + plot_name + ".pdf")







