from dataclasses import dataclass
from typing import Dict, List, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3.math as pmm

from scipy.stats import bernoulli, expon

# PYMC3 ######################################################################
@dataclass
class BetaPrior:
    alpha: float
    beta: float
@dataclass
class BinomialData:
    trials: int
    successes: int

RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)


class ConversionModelTwoVariant:
    def __init__(self, priors: BetaPrior):
        self.priors = priors

    def create_model(self, data: List[BinomialData]) -> pm.Model:
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=2)
            obs = pm.Binomial("y", n=trials, p=p, shape=2, observed=successes)
            reluplift = pm.Deterministic("reluplift_b", p[1] / p[0] - 1)
        return model


def generate_binomial_data(
        variants: List[str],
        true_rates: List[float],
        samples_per_variant: int = 10000
) -> pd.DataFrame:
    data = {}
    np.random.seed(0)
    for variant, p in zip(variants, true_rates):
        data[variant] = np.random.binomial(1, p=p, size=samples_per_variant)
    agg = (
        pd.DataFrame(data)
            .aggregate(["count", "sum"])
            .rename(index={"count": "trials", "sum": "successes"})
    )
    return agg


def run_scenario_twovariant(
        variants: List[str],
        true_rates: List[float],
        samples_per_variant: int,
        our_prior: BetaPrior
        # weak_prior: BetaPrior,
        # strong_prior: BetaPrior,
):
    generated = generate_binomial_data(variants, true_rates, samples_per_variant)
    data = [BinomialData(**generated[v].to_dict()) for v in variants]
    with ConversionModelTwoVariant(priors=our_prior).create_model(data):
        trace_our = pm.sample(draws=50000, return_inferencedata=True, cores=1, chains=2)
    # with ConversionModelTwoVariant(priors=weak_prior).create_model(data):
    #     trace_weak = pm.sample(draws=5000, return_inferencedata=True, cores=1, chains=2)
    # with ConversionModelTwoVariant(priors=strong_prior).create_model(data):
    #     trace_strong = pm.sample(draws=5000, return_inferencedata=True, cores=1, chains=2)

    true_rel_uplift = true_rates[1] / true_rates[0] - 1

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    az.plot_posterior(trace_our.posterior["reluplift_b"], textsize=10, ax=axs[0], kind="hist")
    axs[0].set_title(f"True Rel Uplift = {true_rel_uplift:.1%}, {our_prior}", fontsize=10)
    axs[0].axvline(x=0, color="red")
    # az.plot_posterior(trace_strong.posterior["reluplift_b"], textsize=10, ax=axs[1], kind="hist")
    # axs[1].set_title(f"True Rel Uplift = {true_rel_uplift:.1%}, {strong_prior}", fontsize=10)
    # axs[1].axvline(x=0, color="red")
    fig.suptitle("B vs. A Rel Uplift")
    return data

# CPRIOR ############################################
from cprior.models import BernoulliModel, BernoulliMVTest, BernoulliABTest
from cprior.experiment.base import Experiment

modelA = BernoulliModel(name="control", alpha=1, beta=1)
modelB = BernoulliModel(name="variation", alpha=1, beta=1)

mvtest = BernoulliMVTest({"A": modelA, "B": modelB})


def bayes_conversion_stop_experiment(min_n_samples, max_n_samples, p1, p2,
                                     criterion, criterion_value,
                                     seed):
    experiment = Experiment(name="CTR", test=mvtest,
                            stopping_rule=criterion,
                            epsilon=criterion_value, min_n_samples=min_n_samples, max_n_samples=max_n_samples)

    with experiment as e:
        while not e.termination:
            np.random.seed(seed)
            data_A = np.random.binomial(1, p=p1, size=np.random.randint(100, 200))
            data_B = np.random.binomial(1, p=p2, size=np.random.randint(100, 200))

            e.run_update(**{"A": data_A, "B": data_B})
        # print(e.termination, e.status)
    return e

