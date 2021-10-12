import re
from collections import Counter, defaultdict
from math import lgamma
from random import choices
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.stats import beta
import gc
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from numpy import ndarray
from numba import jit

from src.bootstrap import bootstrap_jit_parallel
from src.ab import get_size_zratio

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
                            [n_boots] * len(revenue_dict.keys())))

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


def calc_prob_between(alphas, betas):
    return g(alphas[0], betas[0], alphas[1], betas[1])


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
    # prob_super0 = np.count_nonzero(difference) / size  # probability superiority for
    expected_losses = (np.sum(difference) / size) * 100
    return expected_losses


class BatchThompson:
    def __init__(self, p_control_percent: float, mde_percent: float, batch_size_share_mu: float, criterion_dict: Dict,
                 method_update_params: str, multiarmed: bool = False):
        p1, mde_test = p_control_percent / 100, -(p_control_percent * mde_percent) / 10000
        p2 = p1 - mde_test
        self.p_array_mu = np.array([p1, p2])
        self.n_arms = len(self.p_array_mu)
        self.n_obs_every_arm = get_size_zratio(p_control_percent, mde_percent, alpha=0.05, beta=0.2)
        self.batch_size_share_mu = batch_size_share_mu
        self.criterion_dict = criterion_dict
        self.multiarmed = multiarmed
        self.alphas, self.betas = np.array([1.0] * self.n_arms)
        self.method_update_params = method_update_params
        self.probability_superiority_array = np.array([0.5, 0.5])
        self.expected_losses = 0
        self.cumulative_observations = np.repeat(0, self.n_arms)
        # # Generating data for historic split
        # np.random.seed(seed)
        # self.data = np.random.binomial(n=[1, 1], p=self.p_array_mu,
        #                                size=(self.n_obs_every_arm, self.n_arms))

        # print(f"Нужно наблюдений в каждую руку для выявления эффекта в классическом АБ-тесте: "
        #       f"{self.n_obs_every_arm}")

    def update_beta_params(self, batch_data: np.array):
        if self.method_update_params == "summation":
            self.alphas += np.nansum(batch_data, axis=0)
            self.betas += np.sum(batch_data == 0, axis=0)
        elif self.method_update_params == "normalization":
            S_list = np.nansum(batch_data, axis=0)  # number of successes in within batch
            F_list = np.sum(batch_data == 0, axis=0)
            M = batch_data.shape[0]
            K = self.n_arms

            adding_alphas = (M / K) * (np.array(S_list) / (np.array(S_list) + np.array(F_list)))
            adding_betas = (M / K) * (1 - np.array(S_list) / (np.array(S_list) + np.array(F_list)))

            adding_alphas = np.nan_to_num(adding_alphas)
            adding_betas = np.nan_to_num(adding_betas)

            self.alphas += adding_alphas
            self.betas += adding_betas
        return self.alphas, self.betas

    def update_prob_super(self, method_calc):
        if method_calc == 'integrating':
            prob_superiority = calc_prob_between(self.alphas, self.betas)
            self.probability_superiority_array = np.array([prob_superiority, 1 - prob_superiority])
        self.expected_losses = expected_loss(self.alphas, self.betas)

    def split_data_historic(self, batch_split_obs: List):
        """
        Split data in every batch iteration
        :param self.cumulative_observations: list with cumulative observations for every arm
        :param batch_split_obs: how many observation we must extract this iter
        :return:
        """
        n_rows, n_cols = np.max(batch_split_obs), self.n_arms
        data_split = np.empty((n_rows, n_cols))
        data_split[:] = np.nan
        for i in range(self.n_arms):
            data_split[:batch_split_obs[i], i] = \
                self.data[self.cumulative_observations[i]: self.cumulative_observations[i] + batch_split_obs[i], i]
        return data_split

    def split_data_random(self, batch_split_obs: np.array, seed: int = 1):
        """

        :param batch_split_obs: size for every arm
        :param seed - seed for random numbers
        :return:
        """
        np.random.seed(seed)
        data_split = np.empty((np.max(batch_split_obs), self.n_arms))
        data_split[:] = np.nan
        p_array = self.p_array_mu \
                  # + np.random.normal(0, self.p_array_mu / 3, size=self.n_arms)
        p_array = np.where(p_array < 0, 0, p_array)
        p_array = np.where(p_array > 1, 1, p_array)
        for i in range(self.n_arms):
            data_split[:batch_split_obs[i], i] = np.random.binomial(n=1, p=p_array[i],
                                                                    size=batch_split_obs[i])
            # data_split[:batch_split_obs[i], i] = [1 if j <= p_array[i] else 0 for j in np.random.random(batch_split_obs[i])]

        return data_split

    def start_experiment(self, seed=1):
        probability_superiority_step_list: List[ndarray] = []  # how share of traffic changes across experiment
        observations_step_list: List[ndarray] = []  # how many observations is cumulated in every step
        self.cumulative_observations = np.repeat(0, self.n_arms)  # how many observations we extract every iter for every arm

        while np.max(self.cumulative_observations) < self.n_obs_every_arm:
            batch_size_share = self.batch_size_share_mu + np.random.normal(0, self.batch_size_share_mu / 3)
            batch_size = batch_size_share * self.n_obs_every_arm * 2
            if batch_size < 2:
                batch_size = 2
            batch_split_obs = (batch_size * np.array(self.probability_superiority_array)).astype(
                np.uint16)  # get number of observations every arm
            self.cumulative_observations += batch_split_obs
            # batch_data = batchT.split_data_historic(self.cumulative_observations=self.cumulative_observations,
            #                                         batch_split_obs=batch_split_obs) # based on earlier generated distr
            batch_data = self.split_data_random(batch_split_obs, seed=seed)  # based on generate batch online

            # Updating all
            self.update_beta_params(batch_data, method="normalization")  # update beta distributions parameters
            self.update_prob_super(method_calc="integrating")  # update probability superiority
            self.k += np.round(self.probability_superiority_array, 0)

            # Append for resulting
            probability_superiority_step_list.append(self.probability_superiority_array)
            observations_step_list.append(batch_split_obs)

            # stopping_criterion = (np.max(self.probability_superiority_array) >= 0.99) | \
            #                      (np.max(self.cumulative_observations) >  self.n_obs_every_arm)
        return np.round(probability_superiority_step_list, 3), observations_step_list


class BatchThompsonMixed(BatchThompson):
    """
    Add k and l - number of winners on previous steps
    """

    def update_criterion_stop(self, criterion_name, criterion_value) -> bool:
        if criterion_name == "probability_superiority":
            return (np.max(self.probability_superiority_array) < criterion_value) & \
                   (np.max(self.cumulative_observations[criterion_name]) < self.n_obs_every_arm * 2)
            # # (np.sum(self.cumulative_observations[criterion_name]) < self.n_obs_every_arm * 2)  # condition with True

    def start_experiment(self, seed=1):
        """
        :param criterion_dict:
        :param multiarmed == False -> Bayesian batched
        :param seed - random seed
        :return:
        """
        self.alphas = np.repeat(1.0, self.n_arms)
        self.betas = np.repeat(1.0, self.n_arms)
        self.probability_superiority_array = np.array([0.5, 0.5])
        self.expected_losses = 0
        self.k = (0, 0)  # number of winners for every step
        self.cumulative_observations = np.repeat(0, self.n_arms)
        crit_name, crit_value = list(self.criterion_dict.keys())[0], \
                                list(self.criterion_dict.keys())[1]
        # probability_superiority_step_list: List[tuple] = []  # how share of traffic changes across experiment
        # observations_step_list: List[int] = []  # how many observations is cumulated in every step
        # expected_loss_step_list: List[int] = []
        # k_list_iter: List[ndarray] = []
        probability_superiority_step_list = []  # how share of traffic changes across experiment
        observations_step_list = []  # how many observations is cumulated in every step
        expected_loss_step_list = []
        k_list_iter = []
        k_array, l_array = np.array([0, 0]), np.array([0, 0])
        k_list_iter = [k_array]
        # if crit_n == "expected_loss":
        #     criterion = np.max(self.expected_losses) < crit_value
        # if crit_n == "prob_super_exp_loss":
        #     criterion = (np.max(self.probability_superiority_array) < crit_value[0]) & \
        #                 (np.max(self.expected_losses) < crit_value[1])
        # if crit_n == "full_one_of_the_arm":
        #     criterion = np.max(self.cumulative_observations) < self.n_obs_every_arm
        iteration: int = 1
        while self.update_criterion_stop(crit_name, crit_value):
            batch_size_share = self.batch_size_share_mu + np.random.normal(0, self.batch_size_share_mu / 3)
            batch_size = batch_size_share * self.n_obs_every_arm * 2
            if batch_size < 2:
                batch_size = 2
            # Add condition from paper1 algo
            if sum(2 ** l_array >= k_array) > 0:  # condition to split 50 / 50
                prob_sup_array = np.array((0.5, 0.5))
                k_array = list(map(lambda x: x + 1, k_array))
            else:
                # prob_sup_array = np.round(np.array(self.probability_superiority_array), 2)  # experiment1
                prob_sup_array = np.round(np.array(self.probability_superiority_array), 0)  # experiment2
                k_array = list(map(lambda x, z: x + z,
                                   k_array, np.uint8(np.round(prob_sup_array))))
            k_list_iter.append(k_array)
            l_array = np.array([k_array[0] - k_array[1], k_array[1] - k_array[0]])
            l_array = np.where(l_array < 0, 0, l_array)
            iteration += 1

            # Choose split
            if self.multiarmed is True:
                batch_split_obs = (batch_size * prob_sup_array).astype(
                    np.uint16)  # get number of observations per every arm
            else:
                batch_split_obs = (batch_size * np.array([0.5] * self.n_arms)).astype(
                    np.uint16)  # get number of observations per every arm
            self.cumulative_observations[crit_name] += batch_split_obs
            # print(np.max(self.cumulative_observations))
            batch_data = self.split_data_random(batch_split_obs, seed=seed)  # based on generate batch online

            # Updating all
            self.update_beta_params(batch_data, method="normalization")  # update beta distributions parameters
            self.update_prob_super(method_calc="integrating")  # update probability superiority

            # Append for resulting
            probability_superiority_step_list.append(self.probability_superiority_array)
            observations_step_list.append(batch_split_obs)
            expected_loss_step_list.append(self.expected_losses)
        probability_winner = np.max(self.probability_superiority_array)
        if probability_winner > crit_value:
            winner = np.argmax(self.probability_superiority_array).item()
        else:
            winner = -1
        intermediate_results = (np.round(probability_superiority_step_list, 3),
                                np.round(expected_loss_step_list, 3),
                                observations_step_list,
                                k_list_iter,
                                np.sum(self.alphas)  # sum of conversions
                                )
        gc.collect()
        return winner, intermediate_results


class BatchThompsonOld(BatchThompson):
    """
    Add k and l - number of winners on previous steps
    """

    def update_criterion_stop(self, criterion_name, criterion_value) -> bool:
        if criterion_name == "probability_superiority":
            # return (np.max(self.probability_superiority_array) < criterion_value) & \
            return np.sum(self.cumulative_observations) < self.n_obs_every_arm * 2
            # # (np.sum(self.cumulative_observations[criterion_name]) < self.n_obs_every_arm * 2)  # condition with True

    def start_experiment(self, seed=1):
        """
        :param criterion_dict:
        :param multiarmed == False -> Bayesian batched
        :param seed - random seed
        :return:
        """
        self.alphas = np.repeat(1.0, self.n_arms)
        self.betas = np.repeat(1.0, self.n_arms)
        self.probability_superiority_array = np.array([0.5, 0.5])
        self.expected_losses = 0
        self.cumulative_observations = np.repeat(0, self.n_arms)
        crit_name, crit_value = list(self.criterion_dict.keys())[0], list(self.criterion_dict.values())[0]
        probability_superiority_step_list: List[ndarray] = []  # how share of traffic changes across experiment
        observations_step_list: List[int] = []  # how many observations is cumulated in every step
        expected_loss_step_list: List[int] = []
        while self.update_criterion_stop(crit_name, crit_value):
            batch_size_share = self.batch_size_share_mu + np.random.normal(0, self.batch_size_share_mu / 3)
            batch_size = batch_size_share * self.n_obs_every_arm * 2
            if batch_size < 2:
                batch_size = 2
            prob_sup_array = np.round(np.array(self.probability_superiority_array), 2)
            # Choose split
            if self.multiarmed is True:
                batch_split_obs = (batch_size * prob_sup_array).astype(np.uint32)  # get number of observations per every arm
            else:
                batch_split_obs = (batch_size * np.array([0.5] * self.n_arms)).astype(np.uint32)  # get number of observations per every arm
            self.cumulative_observations += batch_split_obs
            batch_data = self.split_data_random(batch_split_obs, seed=seed)  # based on generate batch online

            # Updating all
            self.update_beta_params(batch_data)  # update beta distributions parameters
            self.update_prob_super(method_calc="integrating")  # update probability superiority

            # Append for resulting
            probability_superiority_step_list.append(self.probability_superiority_array)
            observations_step_list.append(batch_split_obs)
            expected_loss_step_list.append(self.expected_losses)
        probability_winner = np.max(self.probability_superiority_array)
        if probability_winner > crit_value:
            winner = np.argmax(self.probability_superiority_array).item()
        else:
            winner = -1
        intermediate_results = (np.round(probability_superiority_step_list, 3),
                                np.round(expected_loss_step_list, 3),
                                observations_step_list,
                                np.sum(self.alphas)  # sum of conversions
                                )
        gc.collect()
        return winner, intermediate_results
