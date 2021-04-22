import re
from collections import Counter
from random import choices
from typing import List, Dict

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np
from bootstrapped_ways import bootstrap_jit_parallel

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


def ucb_bootstrapped(revenue_dict: Dict[str, np.array],
                     n_boots: int
                     ) -> str:
    """
    Upper Confidence Bounds with Bootstapped Upper Bounds
    :param revenue_dict: keys - handles, values - historical numpy array of revenues

    :return: action;
    """

    # Find upper bounds with bootstrapped samples
    upper_bounds = list(map(bootstrap_jit_parallel, revenue_dict.values(),
                            [n_boots]*len(revenue_dict.keys())))

    n_action = int(np.argmax(upper_bounds))
    return [*revenue_dict.keys()][n_action]





