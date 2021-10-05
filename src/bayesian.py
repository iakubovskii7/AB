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

from src.mab import *


def calculate_bayesian_probability(num_arms, N, random_seed, a, b):
    """
     Calculated the bayesian probabilities by performing
     sampling N trials according to the provided inputs.

    Args:
        num_arms (int): The number of variations to sample from.
        N: The number of sampling trials.
        random_seed: The seed for random number generator.
        a (list): The alpha parameter of a Beta
        distribution. For multiple arms, this will be a list of
        float values.
        b(list): The beta parameter of a Beta
        distribution. For multiple arms, this will be a list of
        float values.

    Returns:
        Ordered list of floating values indicating the success
        rate of each arm in the sampling procedure.
        Success rate is the number of times that the sampling
        returned the maximum value divided by the total number
        of trials.
    """
    np.random.seed(seed=random_seed)
    sim = np.random.beta(a, b, size=(N, num_arms))
    sim_counts = sim.argmax(axis=1)
    unique_elements, counts_elements = np.unique(sim_counts,
                                                 return_counts=True)
    unique_elements = list(unique_elements)
    counts_elements = list(counts_elements)
    for arm in range(num_arms):
        if arm not in unique_elements:
            counts_elements.insert(arm, 0)
    return counts_elements / sum(counts_elements)


class BayesianConversionTest:
    def __init__(self, p_control_percent: float, mde_percent: float, criterion_dict: Dict, alpha=0.05, beta=0.2):
        p1, mde_test = p_control_percent / 100, -(p_control_percent * mde_percent) / 10000
        p2 = p1 - mde_test
        self.p_array_mu = np.array([p1, p2])
        self.n_arms = self.p_array_mu.shape[0]
        self.criterion_dict = criterion_dict
        self.n_obs_every_arm = get_size_zratio(p_control_percent, mde_percent, alpha=alpha, beta=beta)
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

        winner_dict = {key: [] for key in self.criterion_dict.keys()}
        intermediate_dict = {key: [] for key in self.criterion_dict.keys()}
        np.random.seed(seed)
        data = np.random.binomial(n=[1] * self.n_arms, p=self.p_array_mu, size=(self.n_obs_every_arm, self.n_arms))
        self.update_beta_params(data, "summation")
        self.update_prob_super(method_calc="integrating")
        for crit_name, crit_value in self.criterion_dict.items():
            if crit_name == "probability_superiority":
                if np.max(self.probability_superiority_tuple) > crit_value:
                    winner_dict[crit_name] = np.argmax(self.probability_superiority_tuple)
                else:
                    winner_dict[crit_name] = -1  # it means not winner
            intermediate_dict[crit_name] = (self.probability_superiority_tuple,
                                            self.expected_losses,
                                            self.n_obs_every_arm)
        return winner_dict, intermediate_dict

# results_all[0][1]['probability_superiority'][1]
#
# results_all[1]
# print(winner_dict)
#
#
#
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