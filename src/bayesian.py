from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import beta
import numpy as np

from AB.src.mab import calc_prob_between


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

alphas, betas = [334, 338], [1385 - 334, 1385 - 338]
print(calculate_bayesian_probability(2, 10000, 17, alphas, betas))
print(calc_prob_between(alphas, betas))


# PyMC3

np.random.seed(123)
n_experiments = 1000
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)

with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=theta, observed=data)
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start, chains=15, cores=4)


burnin = 100
chain = trace[burnin:]
pm.traceplot(chain, lines={'theta': theta_real})
pm.plot_posterior(chain)



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
