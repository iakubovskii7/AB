from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta
import numpy as np

from AB.src.mab import calc_prob_between


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


class BayesianTest:
    def __init__(self):