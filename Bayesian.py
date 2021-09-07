from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta
import numpy as np
def calculate_bayesian_probability(num_arms, N, random_seed, a, b):
    """
     Calculated the bayesian probabilities by performing
     sampling N trials according to the provided inputs.

    Args:
        num_arms (int): The number of variations to sample from.
        N: The number of sampling trials.
        random_seed: The seed for random number generator.
        a (list): The alpha parameter of a Beta
        distribution. For multiple arms, this will be a list of
        float values.
        b(list): The beta parameter of a Beta
        distribution. For multiple arms, this will be a list of
        float values.

    Returns:
        Ordered list of floating values indicating the success
        rate of each arm in the sampling procedure.
        Success rate is the number of times that the sampling
        returned the maximum value divided by the total number
        of trials.
    """
    np.random.seed(seed=random_seed)
    sim = np.random.beta(a, b, size=(N, num_arms))
    sim_counts = sim.argmax(axis=1)
    unique_elements, counts_elements = np.unique(sim_counts,
                                                 return_counts=True)
    unique_elements = list(unique_elements)
    counts_elements = list(counts_elements)
    for arm in range(num_arms):
        if arm not in unique_elements:
            counts_elements.insert(arm, 0)
    return counts_elements / sum(counts_elements)
print(calculate_bayesian_probability(2, 10000, 17,
                                     [334, 338], [1385 - 334, 1385 - 338]))


# Calculate probability superiority
#defining the functions used
from functions import calc_prob_between

#This is the known data: impressions and conversions for the Control and Test set
imps_ctrl, convs_ctrl = 1385, 334
imps_test, convs_test = 1385, 338

#here we create the Beta functions for the two sets
a_C, b_C = convs_ctrl+1, imps_ctrl-convs_ctrl+1
beta_C = beta(a_C, b_C)
a_T, b_T = convs_test+1, imps_test-convs_test+1
beta_T = beta(a_T, b_T)

#calculating the lift
lift=(beta_T.mean()-beta_C.mean())/beta_C.mean()

#calculating the probability for Test to be better than Control
prob=calc_prob_between(beta_T, beta_C)

print (f"Test option lift Conversion Rates by {lift*100:2.2f}% with {prob*100:2.1f}% probability.")
#output: Test option lift Conversion Rates by 59.68% with 98.2% probability.



# Grid methods
grid_points=100
heads=6
tosses=9
grid = np.linspace(0, 1, 100)
prior = np.repeat(5, grid_points)
likelihood = stats.binom.pmf(heads, tosses, grid)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / unstd_posterior.sum()


# PyMC3

np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
print(data)

with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=theta, observed=data)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start, chains=15, cores=15)

burnin = 100
chain = trace[burnin:]
pm.traceplot(chain, lines={'theta': theta_real})
pm.plot_posterior(chain)

# bayes factor
coins = 30
heads = 9
y = np.repeat([0, 1], [coins - heads, heads])

with pm.Model() as model_BF:
    p = np.array([0.5, 0.5])
    model_index = pm.Categorical('model_index', p=p)
    m_0 = (4, 8)
    m_1 = (8, 4)
    m = pm.math.switch(pm.math.eq(model_index, 0), m_0, m_1)
    theta = pm.Beta('theta', m[0], m[1])
    y = pm.Bernoulli('y', theta, observed=y)
    trace_BF = pm.sample(5000, chains=15, cores=15)
chain_BF = trace_BF[500:]
pm.traceplot(chain_BF)
plt.show()

pM1 = chain_BF['model_index'].mean()
pM0 = 1 - pM1
BF = (pM0/pM1)*(p[1]/p[0])


mu1 = 82
mu2 = 78
var1 = mu1 * (1 - 0.307)
var2 = mu2 * (1 - 0.274)

cohens = (mu1 - mu2) / np.sqrt((var1 + var2) / 2)
stats.beta.cdf(x=cohens/np.sqrt(2), a=(82 + 78) / 2, b=(267 + 284)/2)

stats.norm.cdf(x=cohens/np.sqrt(2))


tips = sns.load_dataset('tips')
y = tips['tip'].values
idx = pd.Categorical(tips['day']).codes
x = set(tips['day'])

with pm.Model() as comparing_groups:
    means = pm.Normal('means', mu=0, sd=10, shape=len(set(x)))
    sds = pm.HalfNormal('sds', sd=10, shape=len(set(x)))
    y = pm.Normal('y', mu=means[idx], sd=sds[idx], observed=y)
    trace_cg = pm.sample(5000, chains=15, cores=15)
chain_cg = trace_cg[100::]
pm.plot_trace(chain_cg)

pm.sample_ppc()

summar = az.summary(trace_cg)

ppc = pm.sample_posterior_predictive(trace_cg, samples=1000,
                                     model=comparing_groups)

az.ppc
