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
    def __init__(self, p_control: float, mde: float, criterion_dict: Dict, alpha=0.05, beta=0.2, seed=1):
        self.p_array_mu = np.array([p_control, (1 + mde) * p_control])
        self.n_arms = self.p_array_mu.shape[0]
        self.alpha, self.beta = alpha, beta
        self.alphas = np.repeat(1.0, self.n_arms)
        self.bethas = np.repeat(1.0, self.n_arms)
        self.n_obs_every_arm = get_size_zratio(self.p_array_mu[0], self.p_array_mu[1],
                                               alpha=self.alpha, beta=self.beta)
        self.n_obs_every_arm = 1000
        if self.n_obs_every_arm == 0:
            self.n_obs_every_arm = 100

        self.seed = seed
        self.criterion_dict = criterion_dict
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

    def start_experiment(self):
        winner_dict = {key: [] for key in self.criterion_dict.keys()}
        intermediate_dict = {key: [] for key in self.criterion_dict.keys()}
        np.random.seed(self.seed)
        data = np.random.binomial(n=[1] * self.n_arms, p=self.p_array_mu, size=(self.n_obs_every_arm, self.n_arms))
        self.update_beta_params(data, "summation")
        self.update_prob_super(method_calc="integrating")
        for crit_name, crit_value in self.criterion_dict.items():
            if crit_name == "probability_superiority":
                if np.max(self.probability_superiority_tuple) > crit_value:
                    winner_dict[crit_name] = np.argmax(self.probability_superiority_tuple)
                else:
                    winner_dict[crit_name] = "not_winner"
            intermediate_dict[crit_name] = (self.probability_superiority_tuple,
                                            self.expected_losses,
                                            self.n_obs_every_arm)
        return winner_dict, intermediate_dict

