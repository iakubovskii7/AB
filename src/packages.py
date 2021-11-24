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

@dataclass
class GammaPrior:
    alpha: float
    beta: float

@dataclass
class RevenueData:
    visitors: int
    purchased: int
    total_revenue: float


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


class RevenueModel:
    def __init__(self, conversion_rate_prior: BetaPrior, mean_purchase_prior: GammaPrior):
        self.conversion_rate_prior = conversion_rate_prior
        self.mean_purchase_prior = mean_purchase_prior

    def create_model(self, data: List[RevenueData], comparison_method: str) -> pm.Model:
        num_variants = len(data)
        visitors = [d.visitors for d in data]
        purchased = [d.purchased for d in data]
        total_revenue = [d.total_revenue for d in data]

        with pm.Model() as model:
            theta = pm.Beta(
                "theta",
                alpha=self.conversion_rate_prior.alpha,
                beta=self.conversion_rate_prior.beta,
                shape=num_variants,
            )
            lam = pm.Gamma(
                "lam",
                alpha=self.mean_purchase_prior.alpha,
                beta=self.mean_purchase_prior.beta,
                shape=num_variants,
            )
            converted = pm.Binomial(
                "converted", n=visitors, p=theta, observed=purchased, shape=num_variants
            )
            revenue = pm.Gamma(
                "revenue", alpha=purchased, beta=lam, observed=total_revenue, shape=num_variants
            )
            revenue_per_visitor = pm.Deterministic("revenue_per_visitor", theta * (1 / lam))
            theta_reluplift = []
            reciprocal_lam_reluplift = []
            reluplift = []
            for i in range(num_variants):
                if comparison_method == "compare_to_control":
                    comparison_theta = theta[0]
                    comparison_lam = 1 / lam[0]
                    comparison_rpv = revenue_per_visitor[0]
                elif comparison_method == "best_of_rest":
                    others_theta = [theta[j] for j in range(num_variants) if j != i]
                    others_lam = [1 / lam[j] for j in range(num_variants) if j != i]
                    others_rpv = [revenue_per_visitor[j] for j in range(num_variants) if j != i]
                    if len(others_rpv) > 1:
                        comparison_theta = pmm.maximum(*others_theta)
                        comparison_lam = pmm.maximum(*others_lam)
                        comparison_rpv = pmm.maximum(*others_rpv)
                    else:
                        comparison_theta = others_theta[0]
                        comparison_lam = others_lam[0]
                        comparison_rpv = others_rpv[0]
                else:
                    raise ValueError(f"comparison method {comparison_method} not recognised.")
                theta_reluplift.append(
                    pm.Deterministic(f"theta_reluplift_{i}", theta[i] / comparison_theta - 1)
                )
                reciprocal_lam_reluplift.append(
                    pm.Deterministic(
                        f"reciprocal_lam_reluplift_{i}", (1 / lam[i]) / comparison_lam - 1
                    )
                )
                reluplift.append(
                    pm.Deterministic(f"reluplift_{i}", revenue_per_visitor[i] / comparison_rpv - 1)
                )
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


def generate_revenue_data(
    variants: List[str],
    true_conversion_rates: List[float],
    true_mean_purchase: List[float],
    samples_per_variant: int,
) -> pd.DataFrame:
    converted = {}
    mean_purchase = {}
    for variant, p, mp in zip(variants, true_conversion_rates, true_mean_purchase):
        converted[variant] = bernoulli.rvs(p, size=samples_per_variant)
        mean_purchase[variant] = expon.rvs(scale=mp, size=samples_per_variant)
    converted = pd.DataFrame(converted)
    mean_purchase = pd.DataFrame(mean_purchase)
    revenue = converted * mean_purchase
    agg = pd.concat(
        [
            converted.aggregate(["count", "sum"]).rename(
                index={"count": "visitors", "sum": "purchased"}
            ),
            revenue.aggregate(["sum"]).rename(index={"sum": "total_revenue"}),
        ]
    )
    return agg


def run_scenario_value(
    variants: List[str],
    true_conversion_rates: List[float],
    true_mean_purchase: List[float],
    samples_per_variant: int,
    conversion_rate_prior: BetaPrior,
    mean_purchase_prior: GammaPrior,
    comparison_method: str,
) -> az.InferenceData:
    generated = generate_revenue_data(
        variants, true_conversion_rates, true_mean_purchase, samples_per_variant
    )
    data = [RevenueData(**generated[v].to_dict()) for v in variants]
    with RevenueModel(conversion_rate_prior, mean_purchase_prior).create_model(
        data, comparison_method
    ):
        trace = pm.sample(draws=5000, return_inferencedata=True, chains=2, cores=1)

    n_plots = len(variants)
    fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(3 * n_plots, 7), sharex=True)
    for i, variant in enumerate(variants):
        if i == 0 and comparison_method == "compare_to_control":
            axs[i].set_yticks([])
        else:
            az.plot_posterior(
                trace.posterior[f"reluplift_{i}"], textsize=10, ax=axs[i], kind="hist"
            )
        true_rpv = true_conversion_rates[i] * true_mean_purchase[i]
        axs[i].set_title(f"Rel Uplift {variant}, True RPV = {true_rpv:.2f}", fontsize=10)
        axs[i].axvline(x=0, color="red")
    fig.suptitle(f"Method {comparison_method}, {conversion_rate_prior}, {mean_purchase_prior}")

    return trace

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

