import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from AB_classic import param_mean_tests

# DataSets for ARPU
df1 = np.random.exponential(scale=5, size=10000)
sns.displot(df1)
plt.show()

df2 = np.random.exponential(scale=3, size=10000)

df1.mean(), df2.mean()

# Change old data
big_price_before = pd.read_csv("Data/Quantum_AB_historic/ARPU/IOS/Revenue/BigPriceBefore.csv")

shown_discount = big_price_before[big_price_before['group_test'] == 'shown_discount']['revenue']
shown_no_discount = big_price_before[big_price_before['group_test'] == 'shown_no_discount']['revenue']
param_mean_tests(shown_discount, shown_no_discount)

new_shown_discount = shown_discount + np.random.exponential(scale=0.2, size=shown_discount.shape[0])
print(new_shown_discount.mean(), shown_no_discount.mean())
param_mean_tests(new_shown_discount, shown_no_discount)

cumsum_mean1 = new_shown_discount.cumsum() / np.arange(new_shown_discount.shape[0])
cumsum_mean2 = shown_no_discount.cumsum() / np.arange(shown_no_discount.shape[0])

pd.DataFrame(columns=['cumsum1', 'cumsum2'],
             data=np.hstack([cumsum_mean1.values.reshape(-1,1),
                             cumsum_mean2.values.reshape(-1,1)]))
sns.displot(shown_discount, axis=1)
plt.show()

shown_discount.describe()

## ARPU new random
arpu1 = np.random.exponential(scale=2, size=10000)
arpu2 = np.random.exponential(scale=2.1, size=10000)

print(arpu1.mean(), arpu2.mean())
param_mean_tests(arpu1, arpu2)

arpu1 = np.random.exponential(scale=0.2, size=10000)
arpu2 = np.random.exponential(scale=0.21, size=10000)
print(arpu1.mean(), arpu2.mean())
param_mean_tests(arpu1, arpu2)
pd.DataFrame(columns=list("AB"), data = np.hstack([arpu1.reshape(-1,1),
                                                   arpu2.reshape(-1,1)])).to_csv("Data/Simulation_Data/ARPU5.csv")



## Conversion new random (p=0.4)
conv1 = np.random.binomial(n=1, p=0.1, size=10000)
conv2 = np.random.binomial(n=1, p=0.101, size=10000)
print(np.mean(conv1), np.mean(conv2))
param_mean_tests(conv1, conv2)
pd.DataFrame(columns=list("AB"), data = np.hstack([conv1.reshape(-1,1),
                                                   conv2.reshape(-1,1)])).to_csv("Data/Simulation_Data/Conversion5.csv")

## Number of target actions
## Conversion new random (p=0.4)
target1 = np.random.binomial(n=1, p=0.75, size=10000)
target2 = np.random.binomial(n=1, p=0.7, size=10000)
print(np.mean(target1), np.mean(target2))
param_mean_tests(target1, target2)
pd.DataFrame(columns=list("AB"), data = np.hstack([target1.reshape(-1,1),
                                                   target2.reshape(-1,1)])).to_csv("Data/Simulation_Data/TargetActions5.csv")

