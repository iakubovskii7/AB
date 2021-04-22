from __future__ import division
from scipy.stats import beta
from bandits import BernoulliBandit
import numpy as np
import time

class Solver(object):
    def __ini__(self, bandit):
        '''
        bandit (Bandit): the target bandit to solve
        :param bandit:
        :return:
        '''
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))
        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        
