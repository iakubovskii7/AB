from AB_classic import param_mean_tests
import scipy
# Analyse Anna MABs

import pandas as pd
import numpy as np
p1, p2 = 0.6054255319, 0.5879674797
n1, n2 = 18800, 18450
p = (n1*p1 + n2*p2) / (n1 + n2)
z = (p1 - p2) / np.sqrt(p * (1 - p) *(1/n1 + 1/n2))
print(z > 1.96)

# Analyse AB test Quantum
big_price_before = pd.read_csv("Data/Quantum_AB_historic/ARPU/IOS/Revenue/BigPriceBefore.csv")

big_price_before.pivot_table(index=['appsflyer_id'],
                             columns=['group_test'],
                             values=['revenue'])
shown_discount = big_price_before[big_price_before['group_test'] == 'shown_discount']['revenue']
shown_no_discount = big_price_before[big_price_before['group_test'] == 'shown_no_discount']['revenue']
param_mean_tests(shown_discount, shown_no_discount)

n1, n2 = shown_discount.shape[0], shown_no_discount.shape[0]
print(f"observations n1, n2: {n1, n2}")
mean1, mean2 = np.mean(shown_discount), np.mean(shown_no_discount)
print(f"mean1, mean2: {mean1, mean2}")
sd1, sd2 = np.std(shown_discount, ddof=1), np.std(shown_no_discount, ddof=1)
print(f"standard deviation 1, standard deviation 2: {sd1, sd2}")
se1, se2 = sd1 / np.sqrt(n1), sd2 / np.sqrt(n2)
print(f"standard error 1, standard error 2: {se1, se2}")
sed = np.sqrt(se1**2.0 + se2**2.0)
print(f"standard error for statistics: {sed}")
t_stat = (mean1 - mean2) / sed
print(t_stat)

df = n1 + n2 - 2
# calculate p value
alpha = 0.05
p = (1 - scipy.stats.t.cdf(abs(t_stat), df)) * 2

# SkipAds
skip_ads_diamond = pd.read_csv("Data/Quantum_AB_historic/ARPU/IOS/Revenue/SkipAdsByDiamond.csv")
diamond_a = skip_ads_diamond[skip_ads_diamond['group_test'] == 'A']['revenue'].dropna()
diamond_b = skip_ads_diamond[skip_ads_diamond['group_test'] == 'B']['revenue'].dropna()
param_mean_tests(diamond_a, diamond_b)




