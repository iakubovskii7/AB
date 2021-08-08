import re
from collections import Counter
from random import choices
from typing import List, Set, Dict, Tuple, Optional, NamedTuple
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


# Pure realisation for classic MAB

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


# Batch Thompson bandits
class BernoulliBanditBatch:
    def __init__(self, probs: List):
        self._probs = probs
        self._n_actions = len(probs)

    @property
    def action_count(self):
        return len(self._probs)

    def generate_events(self, thetas, batch_size):
        shares_arm = thetas / sum(thetas)
        rewards_generate_dict = {i: np.random.binomial(1, self._probs[i], shares_arm[i] * batch_size)
                                 for i, _ in enumerate(self._probs)}
        return rewards_generate_dict

    def pull_arms(self, rewards_generate_dict, alphas: List, betas: List):
        """
        generate rewards according to updated thethas in previous batch
        Args:
            rewards_generate_dict: generate rewards for events
            alphas: alpha in every batch for every arm
            betas: beta in every batch for every arm
        Returns: generate random trials according every arm batch size
        """
        S, F = [0] * len(self._probs), [0] * len(self._probs)  # assign to zero within every batch
        maximum_len_arm = np.max(list(map(len, rewards_generate_dict.values())))
        for event in range(maximum_len_arm):
            random_event_theta = np.random.beta(alphas, betas)
            action_arm_event = np.argmax(random_event_theta)
            if


    def optimal_reward(self, batch: int) -> float:
        """
        Used for regret calculation for every batch step
        """
        return np.max(self._probs) * len(batch)

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
    def init_actions(self, probs):
        """
        Initialize alphas and betas for every arm in first iteration
        Args: probability list
        Returns: cumulative successes and failures for every arm in event
        """
        self._alphas = [1] * len(probs)  # initialize alpha = 1 for every arm
        self._betas = [1] * len(probs)  # initialize beta = 1 for every arm

    @abstractmethod
    def get_action(self):
        """
        Get current best action
        :rtype: int
        """
        pass

    def update(self, action: int, reward: int):
        """
        Observe reward from every action within batch
        :type action: int
        :type reward: int
        """

    @property
    def name(self):
        return self.__class__.__name__


class ThompsonSamplingAgentBatch(AbstractAgentBatch):
    def get_action(self, alphas: List, betas: List):
        """
        Recalculate thetas after each batch
        :weights: np.random.beta(alphas, betas)
        :return: thethas after every each batch
        """
        thetas = np.random.beta(alphas, betas)
        return thetas

    def pull_arm_after_batch(self):
        ...

    @property
    def name(self):
        return self.__class__.__name__


def get_results(env, agents, n_batches=100, batch_range=[10, 100], n_trials=10):
    # regret_scores = OrderedDict({
    #     agent.name: [0.0 for step in range(n_steps)] for agent in agents
    # })
    # reward_scores = OrderedDict({
    #     agent.name: [tuple(0 for i in range(env.arms)) for step in range(n_steps)] for agent in agents
    # })
    # prob_win_scores = OrderedDict({
    #     agent.name: [0.0 for step in range(n_steps)] for agent in agents
    # })

    for trial in range(n_trials):
        env.reset()
        for agent in agents:
            agent.init_actions(env.probs)
        for i in range(n_batches):
            # Generate size of batch
            batch_size = np.random.uniform(batch_range[0], batch_range[1])
            optimal_reward = env.optimal_reward(batch_size)
            for agent in agents:
                if i == 0:  # first batch we divide into equal proportions
                    thetas = 1 / env.action_count
                else:
                    thetas = agent.get_action(agent.alphas, agent.betas)  # get thethas after every batch
                rewards_generate_dict = env.generate_events(thetas, batch_size)
                S, F = env.pull_arms(thetas, batch_size)  # get S and F for every arm
                agent.update(thetas)  # update alpha, beta params for every arm
                # regret_scores[agent.name][i] += optimal_reward - np.sum(reward)
                # reward_scores[agent.name][i][action] = reward

            env.step()  # change bandit step if it is unstationary

    # for agent in agents:
    #     regret_scores[agent.name] = np.cumsum(regret_scores[agent.name]) / n_trials
    #     for arm in env.arms:
    #         reward_scores[agent.name][arm] = np.cumsum(reward_scores[agent.name][arm])
    #
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







