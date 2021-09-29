import os
from src.ab import *
from tests import *
from src.mab import *
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from collections import Counter
from src.bayesian import *

# if __name__ == "__main__":
#     abtest = ABConversionTest(p_control=0.1, mde=0.1, batch_size_share_mu=None, seed=1)
#     winner_dict, intermediate_df = abtest.start_experiment()


# def experiment(iteration):
#     mabtest = BatchThompsonMixed(p_control=0.1, mde=0.1, batch_size_share_mu=0.01,
#                                  criterion_dict={"probability_superiority": 0.99}, seed=iteration * 300)
#     return mabtest.start_experiment(multiarmed=False)
#
#
# if __name__ == "__main__":
#     results_all = Parallel(n_jobs=-1)(
#         delayed(experiment)(i) for i in tqdm(range(1000)))
#
# winners = [results_all[i][0]['probability_superiority'] for i in range(1000)]
# print(Counter(winners))
def experiment_pure_bayes(iteration):
    bayestest = BayesianConversionTest(p_control=0.4, mde=0.1,
                                       criterion_dict={"probability_superiority": 0.95},
                                       seed=iteration)
    return bayestest.start_experiment()


if __name__ == "__main__":
    results_all = Parallel(n_jobs=-1)(
        delayed(experiment_pure_bayes)(i) for i in tqdm(range(1000)))
winners = np.array([results_all[i][0]['probability_superiority'] for i in range(1000)])
print(Counter(winners))
# print(np.sum(winners > 0.9))

means = []
for _ in range(1000):
    np.random.seed(_)
    data = np.random.binomial([1, 1], [0.4, 0.4], (10000, 2))
    means.append(data.mean(axis=0))


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

expected_loss([65, 32], [1263 - 65, 1084 - 32]) * 100
calc_prob_between([65, 32], [1263 - 65, 1084 - 32])