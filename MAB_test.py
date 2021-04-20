import numpy as np
import pandas as pd
import slots
import matplotlib.pyplot as plt
import os

os.chdir("/Users/iakubovskii/AppBooster/Base/AB_testing/")
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

# UCB
from MAB import gener_random_revenue, ucb, mab_experiment_ucb
utility_dict = {"a": 462, "b": 0, "c": 0, "d": 0}
selections_dict = {"a": 28, "b": 0, "c": 0, "d": 0}
print(ucb(utility_dict, selections_dict))

import json
with open('revenue_dict_6.json') as f:
  revenue_dict_6 = json.load(f)
with open('revenue_dict_7.json') as f:
  revenue_dict_7 = json.load(f)




{key: sum(values) for key, values in revenue_dict_6.items()}
utility_dict_6 = {key: 0 for key in revenue_dict_6.keys()}
selections_dict_6 = {key: 0 for key in revenue_dict_6.keys()}
revenue_experiment = np.zeros((10**4, 2))

for i in range(10000):
  n_action = ucb(utility_dict_6, selections_dict_6)
  selections_dict_6[n_action] += 1
  revenue_iter = gener_random_revenue(revenue_dict_6[n_action])[0]
  utility_dict_6[n_action] += revenue_iter
  revenue_experiment[i, 0] = n_action
  revenue_experiment[i, 1] = revenue_iter
df = pd.DataFrame(revenue_experiment)
df['iter'] = df.index
reslts = df.pivot_table(index = 'iter', columns = 0, values = 1).cumsum().fillna(method = "ffill")
reslts.plot()

