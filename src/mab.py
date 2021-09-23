import re
from collections import Counter
from math import lgamma
from random import choices
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.stats import beta

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from numpy import ndarray
from numba import jit

from AB.src.bootstrap import bootstrap_jit_parallel
from AB.src.ab import get_size_zratio

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
# selections_dist - каждый ключ - ручки, значение сколько раз выбрали руку
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
    return upper_conf


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


# Функции для вычисления вероятности превосходства по точной формуле


@jit
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(num - den)


@jit
def g0(a, b, c):
    return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))


@jit
def hiter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d


def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))


def calc_prob_between(alphas, bethas):
    return g(alphas[0], bethas[0], alphas[1], bethas[1])


class BatchThompson:
    def __init__(self, p_list_mu: List[float], batch_size_share_mu: np.float):
        self.p_array_mu = np.array(p_list_mu)
        self.n_arms = len(p_list_mu)
        self.n_obs_every_arm = get_size_zratio(p_list_mu[0], p_list_mu[1], alpha=0.05, beta=0.2)
        self.batch_size_share_mu = batch_size_share_mu
        self.alphas = np.repeat(1.0, self.n_arms)
        self.bethas = np.repeat(1.0, self.n_arms)
        self.probability_superiority_tuple = (0.5, 0.5)
        self.expected_losses = 0
        self.k = (0, 0) # number of winners for every step

       # Generating data for historic split
        np.random.seed(np.uint16(np.random.random(size=1) * 100).item())
        self.data = np.random.binomial(n=[1,1], p=self.p_array_mu,
                                       size=(self.n_obs_every_arm, self.n_arms))

        # print(f"Нужно наблюдений в каждую руку для выявления эффекта в классическом АБ-тесте: "
        #       f"{self.n_obs_every_arm}")


    def update_beta_params(self, batch_data: np.array, method:str):
        if method == "summation":
            self.alphas += np.nansum(batch_data, axis=0)
            self.bethas += np.sum(batch_data == 0, axis=0)
        elif method == "normalization":
            S_list =  np.nansum(batch_data, axis=0)  # number of successes in within batch
            F_list = np.sum(batch_data == 0, axis=0)
            M = batch_data.shape[0]
            K = self.n_arms

            adding_alphas = (M / K ) * (np.array(S_list) / (np.array(S_list) + np.array(F_list)))
            adding_bethas = (M / K ) * (1 - np.array(S_list) / (np.array(S_list) + np.array(F_list)))

            adding_alphas = np.nan_to_num(adding_alphas)
            adding_bethas = np.nan_to_num(adding_bethas)

            self.alphas += adding_alphas
            self.bethas += adding_bethas
        return self.alphas, self.bethas


    def update_prob_super(self, method_calc) -> Tuple:
        if method_calc == 'integrating':
            prob_superiority =  calc_prob_between(self.alphas, self.bethas)
            self.probability_superiority_tuple = (prob_superiority, 1 - prob_superiority)
        self.expected_losses = expected_loss(self.alphas, self.bethas)


    def split_data_historic(self, cumulative_observations: List, batch_split_obs: List):
        """
        Split data in every batch iteration
        :param cumulative_observations: list with cumulative observations for every arm
        :param batch_split_obs: how many observation we must extract this iter
        :return:
        """
        n_rows, n_cols = np.max(batch_split_obs), self.n_arms
        data_split = np.empty((n_rows, n_cols))
        data_split[:] = np.nan
        for i in range(self.n_arms):
            data_split[:batch_split_obs[i], i] = \
                self.data[cumulative_observations[i] : cumulative_observations[i] + batch_split_obs[i], i]
        return data_split


    def split_data_random(self, batch_split_obs: np.array):
        """

        :param batch_split_obs: size for every arm
        :return:
        """
        data_split = np.empty((np.max(batch_split_obs), self.n_arms))
        data_split[:] = np.nan
        p_array = self.p_array_mu + np.random.normal(0, self.p_array_mu/3, size=self.n_arms)
        p_array = np.where(p_array < 0, 0, p_array)
        p_array = np.where(p_array > 1, 0, p_array)
        for i in range(self.n_arms):
            data_split[:batch_split_obs[i], i] = np.random.binomial(n=1, p=p_array[i],
                                                                    size=batch_split_obs[i])
            # data_split[:batch_split_obs[i], i] = [1 if j <= p_array[i] else 0 for j in np.random.random(batch_split_obs[i])]

        return data_split


    def start_experiment(self):

        probability_superiority_step_list: List[ndarray] = []  # how share of traffic changes across experiment
        observations_step_list: List[ndarray] = []  # how many observations is cumulated in every step

        # Plots
        # folder, file_name = self.experiment_name, str(self.p1) + "_" + str(self.p2)
        cumulative_observations = np.repeat(0, self.n_arms)  # how many observations we extract every iter for every arm

        while np.max(cumulative_observations) < self.n_obs_every_arm:
            batch_size_share = self.batch_size_share_mu + np.random.normal(0, self.batch_size_share_mu / 3)
            batch_size = batch_size_share * self.n_obs_every_arm * 2
            if batch_size < 2:
                batch_size = 2
            batch_split_obs = (batch_size * np.array(self.probability_superiority_tuple)).astype(np.uint16)  # get number of observations every arm
            cumulative_observations += batch_split_obs
            # batch_data = batchT.split_data_historic(cumulative_observations=cumulative_observations,
            #                                         batch_split_obs=batch_split_obs) # based on earlier generated distr
            batch_data = self.split_data_random(batch_split_obs)  # based on generate batch online

            # Updating all
            self.update_beta_params(batch_data, method="normalization")  # update beta distributions parameters
            self.update_prob_super(method_calc="integrating") # update probability superiority
            self.k += np.round(self.probability_superiority_tuple, 0)

            # Append for resulting
            probability_superiority_step_list.append(self.probability_superiority_tuple)
            observations_step_list.append(batch_split_obs)

            # stopping_criterion = (np.max(self.probability_superiority_tuple) >= 0.99) | \
            #                      (np.max(cumulative_observations) >  self.n_obs_every_arm)
        return np.round(probability_superiority_step_list, 3), observations_step_list


def expected_loss(alphas, betas, size=10000):
    """
    Calculate expected losses for beta distribution
    :param size: number of random values
    :param alphas: alpha params
    :param betas: beta params
    :return:
    """
    control_thetas = beta.rvs(alphas[0], betas[0], size=size)
    test_thetas = beta.rvs(alphas[1], betas[1], size=size)
    difference = test_thetas - control_thetas
    difference = np.where(difference < 0, 0, difference)
    prob_super0 = np.count_nonzero(difference) / size  # probability superiority for
    expected_losses = np.sum(difference) / size

    return expected_losses
