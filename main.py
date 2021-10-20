import joblib
import numpy as np
from src.ab import *
from src.bayesian import *
from src.mab import *
from tests import *
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from p_tqdm import p_map
import os
import timeout_decorator
import time
folder_experiment = "Experiment4"

# {"probability_superiority": 0.99}

# Classic AB tests
# 1 minutes - 100 iteration for AB classic with 8 workers

# if __name__ == "__main__":
#     for p_control in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
#         for mde in tqdm(np.linspace(0.01, 0.5, 10)):
#             abtest = ABConversionTest(p_control=p_control, mde=mde)
#             results_all = Parallel(n_jobs=8, verbose=5)(
#                 delayed(abtest.start_experiment)(seed=i)
#                 for i in range(1000))
#             joblib.dump(results_all, f"Experiment results/AB_classic/{folder_experiment}/p_control={p_control}__"
#                                      f"mde={round(mde, 2)}")
# # Bayesian tests (not bandits) - split 50/50, not peeking - calculate at once


@timeout_decorator.timeout(600, timeout_exception=StopIteration, use_signals=False)
def save_results_bayesian(p_control_percent, mde_percent, prob_super,
                          share_observation_optimal_arms):
    test = BayesianConversionTest(p_control_percent=p_control_percent,
                                  mde_percent=mde_percent,
                                  criterion_dict={"probability_superiority": prob_super},
                                  share_observation_optimal_arms=share_observation_optimal_arms)
    results_all = list(p_map(test.start_experiment, [i for i in range(1, 1001)]))
    joblib.dump(results_all, f"Experiment results/Conversion/Bayesian/{folder_experiment}/"
                             f"p_control={p_control_percent}__"
                             f"mde={mde_percent}__"
                             f"prob_super={prob_super}__"
                             f"share_observation_optimal_arms={round(share_observation_optimal_arms, 2)}"
                )

# test = BayesianConversionTest(p_control_percent=1,
#                               mde_percent=10,
#                               criterion_dict={"probability_superiority": 0.9},
#                               share_observation_optimal_arms=0.8)
# test.start_experiment()


# if __name__ == "__main__":
#     for p_control_percent in np.arange(5, 15, 3):
#         for mde_percent in np.arange(0, 11, 3):
#             for prob_super in [0.8, 0.85, 0.9, 0.95, 0.99]:
#                 for share_observation_optimal_arms in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
#                     if os.path.exists(f"Experiment results/Bayesian/{folder_experiment}/"
#                                       f"p_control={p_control_percent}__"
#                                       f"mde={mde_percent}__"
#                                       f"prob_super={prob_super}__"
#                                       f"share_observation_optimal_arms={round(share_observation_optimal_arms, 2)}"):
#                         continue
#                     else:
#                         try:
#                             save_results_bayesian(p_control_percent, mde_percent, prob_super, share_observation_optimal_arms)
#                         except StopIteration:
#                             print("Waiting time exceeds 60 seconds")
#                             continue

# MAB tests (bandits)


# @timeout_decorator.timeout(1200, timeout_exception=StopIteration, use_signals=False)
def save_results_mab(p_control_percent, mde_percent, batch_size_share_mu,
                     prob_super, method_update_params, multi_armed):
    test = BatchThompsonOld(p_control_percent=p_control_percent,
                            mde_percent=mde_percent,
                            batch_size_share_mu=batch_size_share_mu,
                            criterion_dict={"probability_superiority": prob_super},
                            method_update_params=method_update_params,
                            multiarmed=multi_armed)
    # results_all = Parallel(n_jobs=-1, verbose=1)(delayed(test.start_experiment)(seed=i) for i in range(1, 1000))
    results_all = list(p_map(test.start_experiment, range(1, 1001)))
    # joblib.dump(results_all, f"Experiment results/Thompson/{folder_experiment}/p_control={p_control_percent}__"
    #                          f"mde={mde_percent}__"
    #                          f"prob_super={prob_super}__"
    #                          f"batch_size_share_mu={round(batch_size_share_mu, 4)}__"
    #                          f"method_update_params={method_update_params}__"
    #                          f"multi_armed={multi_armed}")
    # print(f"Experiment results/Thompson/{folder_experiment}/p_control={p_control_percent}__"
    #       f"mde={mde_percent}__"
    #       f"prob_super={prob_super}__"
    #       f"batch_size_share_mu={round(batch_size_share_mu, 4)}__"
    #       f"method_update_params={method_update_params}__"
    #       f"multi_armed={multi_armed}")

# test = BatchThompsonOld(p_control_percent=1,
#                         mde_percent=10,
#                         batch_size_share_mu=0.1,
#                         criterion_dict={"probability_superiority": 0.9},
#                         method_update_params="summation",
#                         multiarmed=False)
# test.start_experiment()


# if __name__ == "__main__":
#     for p_control_percent in np.arange(1, 15, 5):
#         for mde_percent in np.arange(10, 50, 10):
#             for prob_super in [0.95]:
#                 for batch_size_share_mu in np.linspace(0.01, 0.1, 10):
#                     for method_update_params in ['summation']:
#                         for multi_armed in [True, False]:
#                             if os.path.exists(f"Experiment results/Conversion/Thompson/{folder_experiment}/p_control={p_control_percent}__"
#                                               f"mde={mde_percent}__"
#                                               f"prob_super={prob_super}__"
#                                               f"batch_size_share_mu={round(batch_size_share_mu, 4)}__"
#                                               f"method_update_params={method_update_params}__"
#                                               f"multi_armed={multi_armed}") is True:
#                                 continue
#                             else:
#                                 try:
#                                     save_results_mab(p_control_percent, mde_percent,
#                                                      batch_size_share_mu, prob_super,
#                                                      method_update_params, multi_armed)
#                                 except StopIteration:
#                                     print("Waiting time exceeds 600 seconds")
#                                     continue
# p_control_percent = 1
# mde_percent = 10
# batch_size_share_mu = 0.1
# method_update_params = "summation"
# multi_armed = True
# prob_super = 0.95
# test = BatchThompsonOld(p_control_percent=p_control_percent,
#                         mde_percent=mde_percent,
#                         batch_size_share_mu=batch_size_share_mu,
#                         criterion_dict={"probability_superiority": prob_super},
#                         method_update_params=method_update_params,
#                         multiarmed=multi_armed)






