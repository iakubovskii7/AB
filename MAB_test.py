import numpy as np
import os
import path
import pandas as pd
import slots
import matplotlib.pyplot as plt
import json
from MAB import (gener_random_revenue, ucb,
                 BernoulliBandit, ThompsonSamplingAgent, EpsilonGreedyAgent, UCBAgent,
                 BernoulliBanditData, ThompsonSamplingAgentData,
                 get_regret, plot_regret)
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
# import seaborn as sns

# strategies = [{'strategy': 'eps_greedy', 'regret': [],
#                'label': '$\epsilon$-greedy ($\epsilon$=0.1)'},
#               {'strategy': 'softmax', 'regret': [],
#                'label': 'Softmax ($T$=0.1)'},
#               {'strategy': 'ucb', 'regret': [],
#                'label': 'UCB1'},
#               {'strategy': 'bayesian', 'regret': [],
#                'label': 'Bayesian bandit'},
#               ]
# for s in strategies:
#     s['mab'] = slots.MAB(num_bandits = 2,
#                              probs = [0.1, 0.2],
#                              hist_payouts = [np.random.uniform(0,1, 100).tolist(),
#                                              np.random.uniform(0,1, 100).tolist()]
#               )
#
# # Run trials and calculate the regret after each trial
# for t in range(10000):
#     for s in strategies:
#         s['mab']._run(s['strategy'])
#         s['regret'].append(s['mab'].regret())
#
# # Pretty plotting
# plt.style.use(['seaborn-poster','seaborn-whitegrid'])
#
# plt.figure(figsize=(15,4))
#
# for s in strategies:
#     plt.plot(s['regret'], label=s['label'])
#
# plt.legend()
# plt.xlabel('Trials')
# plt.ylabel('Regret')
# plt.title('Multi-armed bandit strategy performance (slots)')
# plt.ylim(0,0.2);

# UCB classic

# utility_dict = {"a": 462, "b": 0, "c": 0, "d": 0}
# selections_dict = {"a": 28, "b": 0, "c": 0, "d": 0}
# print(ucb(utility_dict, selections_dict))

# UCB bootstrapped
#
# with open('revenue_dict_6.json') as f:
#   revenue_dict_6 = json.load(f)
# with open('revenue_dict_7.json') as f:
#   revenue_dict_7 = json.load(f)
#
# {key: sum(values) for key, values in revenue_dict_6.items()}
# utility_dict_6 = {key: 0 for key in revenue_dict_6.keys()}
# selections_dict_6 = {key: 0 for key in revenue_dict_6.keys()}
# revenue_experiment = np.zeros((10**4, 2))
#
# for i in range(10000):
#   n_action = ucb(utility_dict_6, selections_dict_6)
#   selections_dict_6[n_action] += 1
#   revenue_iter = gener_random_revenue(revenue_dict_6[n_action])[0]
#   utility_dict_6[n_action] += revenue_iter
#   revenue_experiment[i, 0] = n_action
#   revenue_experiment[i, 1] = revenue_iter
# df = pd.DataFrame(revenue_experiment)
# df['iter'] = df.index
# reslts = df.pivot_table(index = 'iter', columns = 0, values = 1).cumsum().fillna(method = "ffill")
# reslts.plot()

# Uncomment agents
agents = [
     EpsilonGreedyAgent(),
     UCBAgent(),
     ThompsonSamplingAgent()
]
agents = [
         ThompsonSamplingAgentData()
]

experiment_raw = pd.read_csv("Data/Simulation_Data/Conversion2.csv").drop("Unnamed: 0", axis=1)
experiment_raw.columns = np.arange(len(experiment_raw.columns))

regret, reward = get_regret(BernoulliBanditBatch(experiment_raw, n_chunks=50), agents,
                    n_steps=10000/50, n_trials=100)
# regret = get_regret(BernoulliBandit(), agents, n_steps=200, n_trials=100)
plot_regret(agents, regret)
plot_reward(agents, reward)





