import os

import joblib
import numpy as np

from src.ab import *
from tests import *
from src.mab import *
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from collections import Counter
from src.bayesian import *
import gc

# {"probability_superiority": 0.99}

def run_mab_experiment(iteration, p_control, mde, batch_size_share_mu,
                       criterion_bayesian_stop_dict, folder):
    # Classic AB
    mabtest = BatchThompsonMixed(p_control=p_control, mde=mde, batch_size_share_mu=batch_size_share_mu,
                                 criterion_dict=criterion_bayesian_stop_dict, seed=iteration)
    for multiarmed_true_false in [True, False]:
        winner_dict, intermediate_df = mabtest.start_experiment(multiarmed=multiarmed_true_false)
        joblib.dump(winner_dict, f"Thompson/{folder}/mab={multiarmed_true_false}_{p_control}_{mde}_{batch_size_share_mu}"
                                 f"{criterion_bayesian_stop_dict.keys()}="
                                 f"{criterion_bayesian_stop_dict.values()}")
    return

# Classic AB tests
# 1 minutes - 100 iteration for AB classic with 8 workers
folder_experiment = "Experiment1"
# if __name__ == "__main__":
#     for p_control in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
#         for mde in tqdm(np.linspace(0.01, 0.5, 10)):
#             abtest = ABConversionTest(p_control=p_control, mde=mde)
#             results_all = Parallel(n_jobs=8, verbose=5)(
#                 delayed(abtest.start_experiment)(seed=i)
#                 for i in range(1000))
#             joblib.dump(results_all, f"Experiment results/AB_classic/{folder_experiment}/p_control={p_control}_"
#                                      f"mde={round(mde, 2)}")

# # Bayesian tests (not bandits)
# if __name__ == "__main__":
#     for p_control in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
#         for mde in tqdm(np.linspace(0.01, 0.5, 10)):
#             for prob_super in [0.8, 0.85, 0.9, 0.95, 0.99]:
#                 test = BayesianConversionTest(p_control=p_control, mde=mde,
#                                               criterion_dict={"probability_superiority": prob_super})
#                 results_all = Parallel(n_jobs=15, verbose=5)(
#                     delayed(test.start_experiment)(seed=i)
#                     for i in range(1000))
#                 joblib.dump(results_all, f"Experiment results/Bayesian/{folder_experiment}/p_control={p_control}_"
#                                          f"mde={round(mde, 2)}_"
#                                          f"prob_super={prob_super}")

# MAB tests (bandits)
if __name__ == "__main__":
    for p_control in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
        for mde in tqdm(np.linspace(0.01, 0.5, 10)):
            for prob_super in [0.8, 0.85, 0.9, 0.95, 0.99]:
                for multi_armed in [True, False]:
                    test = BatchThompsonMixed(p_control=p_control, mde=mde,
                                              criterion_dict={"probability_superiority": prob_super})
                    results_all = Parallel(n_jobs=15, verbose=5)(
                        delayed(test.start_experiment)(seed=i)
                        for i in range(1000))
                    joblib.dump(results_all, f"Experiment results/Thompson/{folder_experiment}/p_control={p_control}_"
                                             f"mde={round(mde, 2)}_"
                                             f"prob_super={prob_super}")

#
# # winners = [results_all[i][0]['probability_superiority'] for i in range(1000)]
# # print(Counter(winners))
# def experiment_pure_bayes(iteration):
#     bayestest = BayesianConversionTest(p_control=0.4, mde=0.1,
#                                        criterion_dict={"probability_superiority": 0.95},
#                                        seed=iteration)
#     return bayestest.start_experiment()
#
#
# if __name__ == "__main__":
#     results_all = Parallel(n_jobs=-1)(
#         delayed(experiment_pure_bayes)(i) for i in tqdm(range(1000)))
# winners = np.array([results_all[i][0]['probability_superiority'] for i in range(1000)])
# print(Counter(winners))
# print(np.sum(winners > 0.9))


