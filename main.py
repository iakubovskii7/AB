import joblib
import numpy as np
from src.ab import *
from src.bayesian import *
from src.mab import *
from tests import *
from joblib import Parallel, delayed
from p_tqdm import p_map
from tqdm.notebook import tqdm
from collections import Counter
import gc
import os
import time
from pebble import concurrent
import timeout_decorator
folder_experiment = "Experiment1"

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

# # Bayesian tests (not bandits)
# if __name__ == "__main__":
#     for p_control_percent in np.arange(5, 15, 3):
#         for mde_percent in np.arange(0, 11, 3):
#             for prob_super in [0.8, 0.85, 0.9, 0.95, 0.99]:
#                 test = BayesianConversionTest(p_control_percent=p_control_percent,
#                                               mde_percent=mde_percent,
#                                               criterion_dict={"probability_superiority": prob_super})
#                 results_all = Parallel(n_jobs=15, verbose=5)(
#                     delayed(test.start_experiment)(seed=i)
#                     for i in range(1000))
#                 joblib.dump(results_all, f"Experiment results/Bayesian/{folder_experiment}/p_control={p_control_percent}__"
#                                          f"mde={round(mde_percent, 2)}__"
#                                          f"prob_super={prob_super}")

# MAB tests (bandits)


@timeout_decorator.timeout(600, timeout_exception=StopIteration, use_signals=False)
def save_results_combination_params(p_control_percent, mde_percent, batch_size_share_mu,
                                    prob_super, multi_armed):
    test = BatchThompsonMixed(p_control_percent=p_control_percent,
                              mde_percent=mde_percent,
                              batch_size_share_mu=batch_size_share_mu,
                              criterion_dict={"probability_superiority": prob_super},
                              multiarmed=multi_armed)
    # results_all = Parallel(n_jobs=-1, verbose=5)(
    #     delayed(test.start_experiment)(seed=i)
    #     for i in range(1000))
    results_all = list(p_map(test.start_experiment, np.arange(1, 1001)))
    joblib.dump(results_all, f"Experiment results/Thompson/{folder_experiment}/p_control={p_control_percent}__"
                             f"mde={mde_percent}__"
                             f"prob_super={prob_super}__"
                             f"batch_size_share_mu={round(batch_size_share_mu, 4)}__"
                             f"multi_armed={multi_armed}")


if __name__ == "__main__":
    for p_control_percent in tqdm(np.arange(1, 15, 5)):
        for mde_percent in tqdm(np.arange(0, 11, 5)):
            for prob_super in tqdm([0.8, 0.85, 0.9, 0.95, 0.99]):
                for batch_size_share_mu in tqdm(np.linspace(0.01, 0.1, 10)):
                    for multi_armed in tqdm([True, False]):
                        start_time = time.time()
                        if os.path.exists(f"Experiment results/Thompson/{folder_experiment}/p_control={p_control_percent}__"
                                          f"mde={mde_percent}__"
                                          f"prob_super={prob_super}__"
                                          f"batch_size_share_mu={round(batch_size_share_mu, 4)}__"
                                          f"multi_armed={multi_armed}") is True:
                            continue
                        else:
                            try:
                                save_results_combination_params(p_control_percent, mde_percent,
                                                                batch_size_share_mu, prob_super, multi_armed)
                            except StopIteration:
                                print("Waiting time exceeds 10 seconds")
                                continue


