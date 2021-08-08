import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print(calculate_bayesian_probability(2, 700, 17, [2, 4.5], [2, 5.5]))
#
# np.mean([np.random.beta(2, 4.5) for _ in range(10000)])
# np.mean([np.random.beta(2, 5.5) for _ in range(10000)])
a, b = [2, 4.5], [2, 5.5]
np.random.seed(seed=np.random.seed(100))
sim = np.random.beta(a, b, size=(700, 2))
sim_counts = sim.argmax(axis=1)
unique_elements, counts_elements = np.unique(sim_counts,
                                             return_counts=True)
unique_elements = list(unique_elements)
counts_elements = list(counts_elements)





from scipy import stats
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
import pymc3 as pm
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
    trace = pm.sample(1000, step=step, start=start)

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
    trace_BF = pm.sample(5000)
chain_BF = trace_BF[500:]
pm.traceplot(chain_BF)

pM1 = chain_BF['model_index'].mean()
pM0 = 1 - pM1
BF = (pM0/pM1)*(p[1]/p[0])


