from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta
import numpy as np
from src.ab import *
from typing import Dict
from itertools import combinations
from numpy import ndarray
from numba import jit
from math import lgamma

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
    return 1 - g(alphas[0], betas[0], alphas[1], betas[1])


def expected_loss(alphas, betas, size=int(1e6)):
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


def chance_to_beat_control(alphas: ndarray, betas: ndarray, size=int(1e6)) -> float:
    """
    Calculate probability superiority with sampling methods for beta distribution
    :param size: number of random values
    :param alphas: alpha params for all variants, first element - control alpha
    :param betas: beta params for all variants, first element - control beta
    :return: probability superiority for beta distribution
    """
    control_thetas = beta.rvs(alphas[0], betas[0], size=size)
    test_thetas = beta.rvs(alphas[1], betas[1], size=size)
    ctbc = np.mean(test_thetas > control_thetas)[1]  #  chance to beat control
    return ctbc


def chance_to_beat_all(alphas: ndarray, betas: ndarray, size=int(1e6)) -> ndarray:
    """
    Calculate chance to beat other variants
    :param size: number of random values
    :param alphas: alpha params for all variants - first variant we calculate
    :param betas: beta params for all variants first variant we calculate
    :return: probability superiority for conversion tests for first variant
    """

    result = []
    for i in range(alphas.shape[0]):
        target_variant = stats.beta(alphas[i], betas[i]).rvs(size=size)
        distr_all = []
        for j in range(alphas.shape[0]):
            if j != i:
                distr_all.append(stats.beta(alphas[j], betas[j]).rvs(size=size))

        maxall = np.maximum.reduce(distr_all)
        result.append((target_variant > maxall).mean())
    return np.array(result)


def expected_losses_all(alphas: ndarray, betas: ndarray, size=int(1e6)) -> ndarray:
    """
    Calculate chance to beat other variants
    :param size: number of random values
    :param alphas: alpha params for all variants - first variant we calculate
    :param betas: beta params for all variants first variant we calculate
    :return: probability superiority for conversion tests for first variant
    """

    result = []
    for i in range(alphas.shape[0]):
        target_variant = stats.beta(alphas[i], betas[i]).rvs(size=size)
        distr_all = []
        for j in range(alphas.shape[0]):
            if j != i:
                distr_all.append(stats.beta(alphas[j], betas[j]).rvs(size=size))

        maxall = np.maximum.reduce(distr_all)
        result.append(np.maximum(maxall - target_variant, 0).mean())
    return np.array(result)


def expected_related_losses_all(alphas: ndarray, betas: ndarray, size=int(1e6)) -> ndarray:
    """
    Calculate chance to beat other variants
    :param size: number of random values
    :param alphas: alpha params for all variants - first variant we calculate
    :param betas: beta params for all variants first variant we calculate
    :return: probability superiority for conversion tests for first variant
    """

    result = []
    for i in range(alphas.shape[0]):
        target_variant = stats.beta(alphas[i], betas[i]).rvs(size=size)
        distr_all = []
        for j in range(alphas.shape[0]):
            if j != i:
                distr_all.append(stats.beta(alphas[j], betas[j]).rvs(size=size))

        maxall = np.maximum.reduce(distr_all)
        result.append(((maxall - target_variant) / target_variant).mean())
    return np.array(result)


def hdi_mc(x, interval_length=0.9) -> ndarray:
    """
    Compute highest density credible intervals
    :param x:  any distribution params or difference
    :param interval_length: width for credible intervals
    :return: HDI numpy array
    """
    n = len(x)
    xsorted = np.sort(x)
    n_included = int(np.ceil(interval_length * n))
    n_ci = n - n_included
    ci = xsorted[n_included:] - xsorted[:n_ci]
    j = np.argmin(ci)
    hdi_min = xsorted[j]
    hdi_max = xsorted[j + n_included]
    return np.array([hdi_min, hdi_max])


class BayesianConversionTest:
    def __init__(self, p_control_percent: float, mde_percent: float, criterion_dict: Dict,
                 share_observation_optimal_arms=1.0, alpha=0.05, beta=0.2):
        p1, mde_test = p_control_percent / 100, -(p_control_percent * mde_percent) / 10000
        p2 = p1 - mde_test
        self.p_array_mu = np.array([p1, p2])
        self.n_arms = self.p_array_mu.shape[0]
        self.criterion_dict = criterion_dict
        self.n_obs_every_arm = int(get_size_zratio(p_control_percent, mde_percent, alpha=alpha, beta=beta) * \
                                   share_observation_optimal_arms)
        self.alphas = np.repeat(1.0, self.n_arms)
        self.bethas = np.repeat(1.0, self.n_arms)
        self.probability_superiority_tuple = (0.5, 0.5)
        self.expected_losses = (0, 0)
    # TODO: expand for multiple testing case

    def update_beta_params(self, batch_data: np.array, method: str):
        if method == "summation":
            self.alphas += np.nansum(batch_data, axis=0)
            self.bethas += np.sum(batch_data == 0, axis=0)
        elif method == "normalization":
            S_list = np.nansum(batch_data, axis=0)  # number of successes in within batch
            F_list = np.sum(batch_data == 0, axis=0)
            M = batch_data.shape[0]
            K = self.n_arms

            adding_alphas = (M / K) * (np.array(S_list) / (np.array(S_list) + np.array(F_list)))
            adding_bethas = (M / K) * (1 - np.array(S_list) / (np.array(S_list) + np.array(F_list)))

            adding_alphas = np.nan_to_num(adding_alphas)
            adding_bethas = np.nan_to_num(adding_bethas)

            self.alphas += adding_alphas
            self.bethas += adding_bethas
        return self.alphas, self.bethas

    def update_prob_super(self, method_calc):
        if method_calc == 'integrating':
            prob_superiority = calc_prob_between(self.alphas, self.bethas)
            self.probability_superiority_tuple = (prob_superiority, 1 - prob_superiority)
        self.expected_losses = expected_loss(self.alphas, self.bethas)

    def start_experiment(self, seed=1):
        self.alphas = np.repeat(1.0, self.n_arms)
        self.bethas = np.repeat(1.0, self.n_arms)
        self.probability_superiority_tuple = (0.5, 0.5)
        self.expected_losses = (0, 0)
        np.random.seed(seed)

        data = np.random.binomial(n=[1] * self.n_arms, p=self.p_array_mu, size=(self.n_obs_every_arm, self.n_arms))
        self.update_beta_params(data, "summation")
        self.update_prob_super(method_calc="integrating")
        crit_name, crit_value = list(self.criterion_dict.keys())[0], list(self.criterion_dict.values())[0]
        winner = -1
        if crit_name == "probability_superiority":
            if np.max(self.probability_superiority_tuple) > crit_value:
                winner = np.argmax(self.probability_superiority_tuple).item()
        intermediate_results = (self.probability_superiority_tuple,
                                self.expected_losses,
                                self.n_obs_every_arm)
        return winner, intermediate_results

# import math
#
# def calc_ab(alpha_a, beta_a, alpha_b, beta_b):
#     '''
#     See http://www.evanmiller.org/bayesian-ab-testing.html
#     αA is one plus the number of successes for A
#     βA is one plus the number of failures for A
#     αB is one plus the number of successes for B
#     βB is one plus the number of failures for B
#     '''
#     total = 0.0
#     for i in range(alpha_b):
#         num = math.lgamma(alpha_a+i) + math.lgamma(beta_a+beta_b) + math.lgamma(1+i+beta_b) + math.lgamma(alpha_a+beta_a)
#         den = math.log(beta_b+i) + math.lgamma(alpha_a+i+beta_a+beta_b) + math.lgamma(1+i) + math.lgamma(beta_b) + math.lgamma(alpha_a) + math.lgamma(beta_a)
#
#         total += math.exp(num - den)
#     return total
#
# print(calc_ab(1600+1,1500+1,3200+1,3300+1))

# expected_loss([65, 32], [1263 - 65, 1084 - 32]) * 100
# calc_prob_between([65, 32], [1263 - 65, 1084 - 32])