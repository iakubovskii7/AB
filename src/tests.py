import pandas as pd
from AB.src.ab import get_size_zratio, get_size_student, get_bs_confidence_interval
from joblib import Parallel, delayed
import numpy as np

class AlgoTestConversion:
    """
    Class for testing three types of algorithms: AB classic, Bayesian AB, Thompson
    :param algorithm (AB, Bayes, Thompson)
    :param conversion = False -> ARPU
    :param **params depend on metric (type of distribution, control metric, mde, standard deviation)
    """

    def __init__(self, algorithm: classmethod, *params):
        self.p_control, self.mde, self.batch_size_share = params
        if self.conversion is True:
            self.n_obs_every_arm = get_size_zratio(self.p_control, (1 + self.mde) * self.p_control,
                                                   alpha=0.05, beta=0.2)
        self.algorithm = algorithm()

    def one_iteration(self, iter):
        """
        :param iter: iteration for every algo
        :return:
        """
        np.random.seed(np.random.random(iter) * 100)
        if
        if self.distr_type == "normal":
            data = np.random.normal(means, stds, size=(size, len(means)))
        if self.distr_type == "lognormal":
            data = np.random.normal(means, stds, size=(size, len(means)))
        if self.distr_type == "binomial":
            data = np.random.binomial(n=[1, 1], p=np.abs(1 / (1 + means)),
                                      size=(size, means.shape[0]))
        algo = self.algorithm()

    # TODO: add intermediate results
    def start_testing(self, n_iterations: int = 1000):
        winner_indices = Parallel(n_jobs=-1)(delayed(self.algorithm.start_experiment)()
                                             for _ in range(n_iterations)
                                             )
        return winner_indices
