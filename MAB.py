from collections import Counter
from random import choices
from typing import List, Dict
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import re
import numpy as np
import pandas as pd


def gener_random_revenue(revenue_values: List):
    """
    :param revenue_values: historical revenue values
    :return: random value according to historical distribution
    """
    revenue_counter = Counter(revenue_values)
    values = list(revenue_counter.keys())
    probs = list(map(lambda x: x / len(revenue_values), list(revenue_counter.values())))
    return choices(values, probs)

# utility_dict - накопленные суммы по каждой ручки (reward)
# selections_dist - кажыдй ключ - ручки, значение сколько раз выбрали руку
# explore_coefficient = 2 (есть разные эвристики); важный для expl-expl - исследуешь и узнаешь; с 2 все круто и можно уменьшать
# могу уменьшать прямо во время эксперимента (с каждым шагом уменьшаем на 1%) с каждым шагом должен уменьшаться на log() -

def ucb(utility_dict: Dict[str, float],
        selections_dict: Dict[str, int],
        explore_coefficient: float = 2.0
        ) -> str:
    """
    Upper Confidence Bounds
    :param utility_dict: keys - handles, values - performance score;
    :param selections_dict: keys - handles, values - count of selections some handle;
    :param explore_coefficient: heuristic value for bound;

    :return: action;
    """
    rewards = np.fromiter(utility_dict.values(), dtype=float) / (np.fromiter(selections_dict.values(), dtype=float)
                                                                 + 1e-05)
    selections = np.fromiter(selections_dict.values(), dtype=float)
    if sum(selections) != 0:
        sel_sum_log = np.log(sum(selections))
    else:
        sel_sum_log = 0
    n_action = int(np.argmax(
        rewards + explore_coefficient * np.sqrt(sel_sum_log / (selections + 1e-5))
    ))
    return [*utility_dict.keys()][n_action]

def get_bootstrap_upper_bound(revenues: np.array) -> float:
    """

    :param rewards:
    :return: 95% upper bounds
    """
    revenues = revenues + 1e-05
    bootstr_mean = bs.bootstrap(stat_func=bs_stats.mean,
                                values=revenues,
                                num_threads=1,
                                num_iterations=1e04)
    conf_int = re.findall(r"\(.*", str(bootstr_mean))[0]
    upper_conf = float(conf_int.strip(")").split(", ")[1])
    return(upper_conf)

def ucb_bootstrapped(revenue_dict: Dict[str, list]
        ) -> str:
    """
    Upper Confidence Bounds with Bootstapped Upper Bounds
    :param revenue_dict: keys - handles, values - historical list of revenues

    :return: action;
    """
    mean_revenue = np.fromiter(list(map(np.mean, revenue_dict.values())), dtype=float)
    # Find upper bounds with bootstrapped samples
    upper_bounds = list(map(lambda x: get_bootstrap_upper_bound(x),
                            list(map(np.array, revenue_dict.values()))))

    n_action = int(np.argmax(mean_revenue + upper_bounds))
    return [*revenue_dict.keys()][n_action]

def mab_experiment_ucb_bootstrap(revenue_dict_mab: Dict[str, int],
                       iterations: int) -> pd.DataFrame:
    """
    :param revenue_dict_mab: historical revenue for every arm to extract random values
    :param iterations: number of iteration for MAB experiment
    :return: data frame with cumulative revenue for every iteration for every arm
    """
    utility_dict_mab = {key: 0 for key in revenue_dict_mab.keys()}
    selections_dict_mab = {key: 0 for key in revenue_dict_mab.keys()}
    revenue_experiment = np.zeros((iterations, 2))
    for i in range(iterations):
        n_action = ucb_bootstrapped(revenue_experiment, selections_dict_mab)
        selections_dict_mab[n_action] += 1
        revenue_iter = gener_random_revenue(revenue_dict_mab[n_action])[0]
        utility_dict_mab[n_action] += revenue_iter
        revenue_experiment[i, 0] = n_action
        revenue_experiment[i, 1] = revenue_iter
    summarize_df = pd.DataFrame(revenue_experiment)
    summarize_df['iter'] = summarize_df.index
    reslts = summarize_df.pivot_table(index='iter', columns=0, values=1).cumsum().fillna(method="ffill")
    return reslts

def mab_experiment_ucb(revenue_dict_mab: Dict[str, int],
                       iterations: int) -> pd.DataFrame:
    """
    :param revenue_dict_mab: historical revenue for every arm to extract random values
    :param iterations: number of iteration for MAB experiment
    :return: data frame with cumulative revenue for every iteration for every arm
    """
    utility_dict_mab = {key: 0 for key in revenue_dict_mab.keys()}
    selections_dict_mab = {key: 0 for key in revenue_dict_mab.keys()}
    revenue_experiment = np.zeros((iterations, 2))
    for i in range(iterations):
        n_action = ucb(utility_dict_mab, selections_dict_mab)
        selections_dict_mab[n_action] += 1
        revenue_iter = gener_random_revenue(revenue_dict_mab[n_action])[0]
        utility_dict_mab[n_action] += revenue_iter
        revenue_experiment[i, 0] = n_action
        revenue_experiment[i, 1] = revenue_iter
    summarize_df = pd.DataFrame(revenue_experiment)
    summarize_df['iter'] = summarize_df.index
    reslts = summarize_df.pivot_table(index='iter', columns=0, values=1).cumsum().fillna(method="ffill")
    return reslts
# An example that shows how to use the UCB1 learning policy
# to make decisions between two arms based on their expected rewards.

# # Import MABWiser Library
# from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
#
# # Data
# arms = ['Arm1', 'Arm2']
# decisions = ['Arm1', 'Arm1', 'Arm2', 'Arm1']
# rewards = [20, 17, 25, 9]
#
# # Model
# mab = MAB(arms, LearningPolicy.UCB1(alpha=1.25))
#
# # Train
# mab.fit(decisions, rewards)
#
# # Test
# mab.predict()




