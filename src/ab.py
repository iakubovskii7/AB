from itertools import combinations
import statsmodels.stats.api as sms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from src.bootstrap import bootstrap_jit_parallel
from statsmodels.stats.proportion import proportions_ztest
import joblib
import gc
from typing import List, Tuple


def cohen_size_proportion(p1, p2):
    return np.abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def cohen_size_arpu(mean1, mean2, sd):
    return np.abs(mean1 - mean2) / sd


def get_size_student(mean1, mean2, alpha=0.05, beta=0.2, sd_coef=1):
    sd = mean1 * sd_coef
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    n = (np.sqrt(2) * sd * (z_beta + z_alpha) / (mean1 - mean2)) ** 2
    return np.uint32(n)


def get_size_zratio(p_control_percent, mde_percent, alpha=0.05, beta=0.2, type="equivalence"):
    """
    :param p_control_percent: conversion rate in percent (10 means 10% conversion)
    :param mde_percent: mde in percent
    :param alpha: error I
    :param beta: error II
    :param type: hypothesis
    :return:
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    # Rewrite to notice book 2007 year (SAMPLE SIZE CALCULATION FOR COMPARING PROPORTIONS)
    p1, mde_test = p_control_percent / 100, -(p_control_percent * mde_percent) / 10000
    p2 = p1 - mde_test
    if mde_percent == 0:
        n = 10000
    # TODO: add three types of hypothesis
    elif type == "equivalence":
        n = ((z_alpha + z_beta) ** 2 / (p1 - p2) ** 2) * (p1 * (1 - p1) + p2 * (1-p2))
    return np.uint32(n)


def normality_tests(data, alpha=0.05):
    """
    JB and Shapiro-Wilk tests for normality
    :param data: series
           alpha: significance level
    :return: p-values for JB and SW
    """
    stat_jb, p_jb = jarque_bera(data)
    stat_shapiro, p_shapiro = shapiro(data)
    print('stat_shapiro=%.3f, p_shapiro=%.3f' % (stat_shapiro, p_shapiro))
    if p_shapiro > alpha:
        print('Согласно тесту Шапиро-Уилка на 5% уровне значимости делаем вывод \
    в пользу не отвержения нулевой гипотезы - распределение нормальное')
    else:
        print('Согласно тесту Шапиро-Уилка на 5% уровне значимости делаем вывод \
    в пользу отвержения нулевой гипотезы \ - распределение НЕ нормальное')
    print("\n")
    print('stat_jb=%.3f, p_jb=%.3f' % (stat_jb, p_jb))
    if p_jb > alpha:
        print('Согласно тесту Харке-Бера на 5% уровне значимости делаем вывод \
    в пользу не отвержения нулевой гипотезы - распределение нормальное')
    else:
        print('Согласно тесту Харке-Бера на 5% уровне значимости делаем вывод \
    в пользу отвержения нулевой гипотезы \ - распределение НЕ нормальное')
    print("\n")
    return p_jb, p_shapiro

# FWER — family-wise error rate


def bonferroni_correction_function(rvs, alpha, number_tests):
    """
    Bonferroni correction
    :param rvs:
    :param alpha:
    :param number_tests:
    :return:
    """
    alpha_bonferroni = alpha / number_tests
    counter = 0
    for i in range(number_tests):
        rvs_random = stats.norm.rvs(loc=5, scale=10, size=1000, random_state=i + 1)

        statistic, pvalue = stats.ttest_ind(rvs, rvs_random, equal_var=False)

        if pvalue <= alpha_bonferroni:
            counter = counter + 1

    print(counter)


def bonferroni_holm_correction_function(rvs, alpha, number_tests):
    """
    Bonferroni-Holm correction
    :param rvs:
    :param alpha:
    :param number_tests:
    :return:
    """
    pvalue_test = []
    for i in range(number_tests):
        rvs_random = stats.norm.rvs(loc=5, scale=10, size=1000, random_state=i + 1)

        statistic, pvalue = stats.ttest_ind(rvs, rvs_random, equal_var=False)
        pvalue_test.append(pvalue)

    pvalue_test_sorted = sorted(pvalue_test, key=float)

    counter = 0
    for i in range(number_tests):
        if pvalue_test_sorted[i] <= alpha / (number_tests - i):
            counter = counter + 1

    print(counter)


def sidak_correction_function(rvs, alpha, number_tests):
    FWER = 1 - (1 - alpha) ** (1 / number_tests)
    alpha_sidak = 1 - (1 - FWER) ** (1 / number_tests)

    counter = 0
    for i in range(number_tests):
        rvs_random = stats.norm.rvs(loc=5, scale=10, size=1000, random_state=i + 1)

        statistic, pvalue = stats.ttest_ind(rvs, rvs_random, equal_var=False)

        if pvalue <= alpha_sidak:
            counter = counter + 1

    print(counter)


def get_bs_confidence_interval(data, alpha=0.05):
    quantile_array = np.quantile(data, [alpha/2, 1 - alpha/2])
    return quantile_array


def create_confidence_plot(df_results, directory="Plot/ABClassic"):

    for lower, upper, y in zip(df_results['lower'], df_results['upper'], range(len(df_results))):
        plt.plot((lower, upper), (y, y), 'ro-', color='orange');
        plt.yticks(range(len(df_results)), list(df_results.index));
        plt.axvline(x=0, color='b', ls='--');
        plt.savefig(directory)


class ABTest:
    def __init__(self, data: np.array, alpha=0.05, beta=0.2, equal_var=False):
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.equal_var = equal_var
        self.__all_comparisons_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(list(combinations(np.arange(self.data.shape[1]), 2)),
                                            names=['var1', 'var2']),
            columns=['statistic', 'p_value'])
        """
        Input - numpy array: shape[n_observation; n_variants]
        :param equal_var: assumption about variance
        :param data: np.array where shape[1] == number of potential variants
        :param alpha: significance level
        :param beta: type II error (1 - power of test)
        """

    def check_normality(self) -> pd.DataFrame:
        """
        check normality tests
        :rtype: np.array[
        """
        stat_shapiro, p_shapiro = np.apply_along_axis(shapiro, 0, self.data)
        stat_jarque, p_jarque = np.apply_along_axis(jarque_bera, 0, self.data)
        result_norm = pd.DataFrame(index=np.arange(self.data.shape[1]))
        result_norm['stat_shapiro'] = stat_shapiro.T
        result_norm['p_shapiro'] = p_shapiro.T
        result_norm['stat_jarque_bera'] = stat_jarque.T
        result_norm['p_jarque_bera'] = p_jarque.T
        return result_norm

    def student_multiple_test(self) -> Tuple[pd.DataFrame, any]:
        """
        Student test for independent two samples
        :return: pandas dataframe with results
        """
        all_comparisons_student_df = self.__all_comparisons_df.copy()
        for index, row in all_comparisons_student_df.iterrows():
            all_comparisons_student_df.loc[index, "diff_mean"] = self.data[:, index[0]].mean() - self.data[:,
                                                                                                 index[1]].mean()
            stat_test, p_value = ttest_ind(self.data[:, index[0]], self.data[:, index[1]], equal_var=self.equal_var)
            all_comparisons_student_df.loc[index, "statistic"] = stat_test
            all_comparisons_student_df.loc[index, "p_value"] = p_value

        all_comparisons_student_df.sort_values(['p_value'], inplace=True)
        all_comparisons_student_df['i'] = np.arange(all_comparisons_student_df.shape[0]) + 1
        all_comparisons_student_df['alpha_correction'] = (all_comparisons_student_df['i'] * self.alpha) / \
                                                          all_comparisons_student_df.shape[0]
        all_comparisons_student_df['stat_significance'] = np.where(all_comparisons_student_df['p_value'] >
                                                                   all_comparisons_student_df['alpha_correction'],
                                                                   False, True)
        # Create confident intervals for difference with correction significance level
        uservar = 'equal' if self.equal_var == True else 'unequal'
        for index, row in all_comparisons_student_df.iterrows():
            cm = sms.CompareMeans(sms.DescrStatsW(self.data[:, index[0]]), sms.DescrStatsW(self.data[:, index[1]]))
            all_comparisons_student_df.loc[index, "ci_lower"] = \
            cm.tconfint_diff(usevar=uservar, alpha=row['alpha_correction'])[0]
            all_comparisons_student_df.loc[index, "ci_upper"] = \
            cm.tconfint_diff(usevar=uservar, alpha=row['alpha_correction'])[1]

        # Determine winners
        for index, row in all_comparisons_student_df.iterrows():
            all_comparisons_student_df.loc[index, "winner"] = np.where((row['stat_significance'] is True) &
                                                                       (row['statistic'] < 0),
                                                                       str(index[1]),
                                                                       np.where(row['stat_significance'] is True,
                                                                                str(index[0]), "not_winner")).item()
        winner_count_student = all_comparisons_student_df['winner'].value_counts()

        winner_student = None
        if np.all(np.array(winner_count_student.index)) == "not_winner":
            winner_student = "not_winner"

        return all_comparisons_student_df, winner_count_student

    def mann_whitney_multiple_test(self) -> Tuple[pd.DataFrame, any]:
        """
        Student test for independent two samples
        :return: pandas dataframe with results
        """
        all_comparisons_mannwhitney_df = self.__all_comparisons_df.copy()
        for index, row in all_comparisons_mannwhitney_df.iterrows():
            all_comparisons_mannwhitney_df.loc[index, "diff_mean"] = self.data[:, index[0]].mean() - \
                                                                     self.data[:, index[1]].mean()
            stat_test, p_value = mannwhitneyu(self.data[:, index[0]], self.data[:, index[1]])
            all_comparisons_mannwhitney_df.loc[index, "statistic"] = stat_test
            all_comparisons_mannwhitney_df.loc[index, "p_value"] = p_value

        all_comparisons_mannwhitney_df.sort_values(['p_value'], inplace=True)
        all_comparisons_mannwhitney_df['i'] = np.arange(all_comparisons_mannwhitney_df.shape[0]) + 1
        all_comparisons_mannwhitney_df['alpha_correction'] = (all_comparisons_mannwhitney_df['i'] * self.alpha) / \
                                                              all_comparisons_mannwhitney_df.shape[0]
        all_comparisons_mannwhitney_df['stat_significance'] = np.where(all_comparisons_mannwhitney_df['p_value'] >
                                                                       all_comparisons_mannwhitney_df[
                                                                           'alpha_correction'], False, True)
        # Determine winners
        for index, row in all_comparisons_mannwhitney_df.iterrows():
            all_comparisons_mannwhitney_df.loc[index, "winner"] = np.where((row['stat_significance'] is True) &
                                                                           (row['statistic'] < 0),
                                                                           str(index[1]),
                                                                           np.where(row['stat_significance'] is True,
                                                                                    str(index[0]), "not_winner")).item()
        winner_count_mannwhitney = all_comparisons_mannwhitney_df['winner'].value_counts()

        winner = None
        if np.all(np.array(winner_count_mannwhitney.index)) == "not_winner":
            winner_student = "not_winner"

        return all_comparisons_mannwhitney_df, winner_count_mannwhitney

    def bootstrap_multiple_test(self, n_boots: int) -> Tuple[pd.DataFrame, any]:
        all_comparisons_bootstrap_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(list(combinations(np.arange(self.data.shape[1]), 2)),
                                            names=['var1', 'var2']),
            columns=['bs_difference_means', 'p_value', 'bs_confident_interval'])
        # Calculate bs samples
        for index, row in all_comparisons_bootstrap_df.iterrows():
            data1_bs_sample_means = bootstrap_jit_parallel(self.data[:, index[0]], n_boots=n_boots)
            data2_bs_sample_means = bootstrap_jit_parallel(self.data[:, index[1]], n_boots=n_boots)
            difference_bs_means = data1_bs_sample_means - data2_bs_sample_means
            all_comparisons_bootstrap_df.at[index, "bs_difference_means"] = difference_bs_means
            all_comparisons_bootstrap_df.loc[index, "mean1_mean2_diff"] = np.mean(data1_bs_sample_means) - np.mean(data2_bs_sample_means)
            all_comparisons_bootstrap_df.loc[index, "p_value"] = 2 * np.min([np.sum(difference_bs_means < 0) / n_boots,
                                                                         1 - np.sum(difference_bs_means < 0) / n_boots])
        all_comparisons_bootstrap_df.sort_values(['p_value'], inplace=True)
        all_comparisons_bootstrap_df.loc[:, 'i'] = np.arange(all_comparisons_bootstrap_df.shape[0]) + 1
        all_comparisons_bootstrap_df.loc[:, 'alpha_correction'] = (all_comparisons_bootstrap_df['i'] * self.alpha) / \
                                                                all_comparisons_bootstrap_df.shape[0]

        # Create confident intervals for difference with correction significance level
        for index, row in all_comparisons_bootstrap_df.iterrows():
            all_comparisons_bootstrap_df.at[index, 'bs_confident_interval'] = get_bs_confidence_interval(
                all_comparisons_bootstrap_df.loc[index, "bs_difference_means"],
                alpha=all_comparisons_bootstrap_df.loc[index, 'alpha_correction'])
        all_comparisons_bootstrap_df["ci_lower"] = all_comparisons_bootstrap_df.loc[:, "bs_confident_interval"].apply(
            lambda x: x[0])
        all_comparisons_bootstrap_df["ci_upper"] = all_comparisons_bootstrap_df.loc[:, "bs_confident_interval"].apply(
            lambda x: x[1])

        all_comparisons_bootstrap_df['stat_significance'] = np.where(
            (all_comparisons_bootstrap_df['ci_lower'] > 0) |
            (all_comparisons_bootstrap_df['ci_upper'] < 0), True, False)

        # Determine winners
        for index, row in all_comparisons_bootstrap_df.iterrows():
            all_comparisons_bootstrap_df.loc[index, "winner"] = np.where(
                (row['stat_significance'] is True) &
                (row['mean1_mean2_diff'] < 0), str(index[1]), np.where(
                    row['stat_significance'] is True, str(index[0]), "not_winner")).item()
        winner_count_bootstrap = all_comparisons_bootstrap_df['winner'].value_counts()

        winner = None
        if np.all(np.array(winner_count_bootstrap.index)) == "not_winner":
            winner = "not_winner"

        return all_comparisons_bootstrap_df, winner_count_bootstrap

    def start_experiment(self, n_boots=10000):
        all_comparisons_student_df, winner_count_student = self.student_multiple_test()
        all_comparisons_mannwhitney_df, winner_count_mannwhitney = self.mann_whitney_multiple_test()
        all_comparisons_bootstrap_df, winner_count_bootstrap = self.bootstrap_multiple_test(n_boots=n_boots)
        result_df = all_comparisons_student_df.join(all_comparisons_bootstrap_df,
                                                    lsuffix='_Student', rsuffix='_bootstrap').join(
            all_comparisons_mannwhitney_df, rsuffix='_Mann_Whitney'
        )
        return result_df.T


class ABConversionTest:
    def __init__(self, p_control: float, mde: float, alpha: float = 0.05, beta: float = 0.2):
        self.p_control = p_control
        self.mde = mde
        self.p_array_mu = np.array([p_control / 100, self.p_control / 100 + (self.p_control * self.mde) / 10000])
        self.n_arms = self.p_array_mu.shape[0]
        self.alpha, self.beta = alpha, beta
        self.n_obs_every_arm = get_size_zratio(self.p_control, mde,
                                               alpha=self.alpha, beta=self.beta)
        self.__all_comparisons_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(list(combinations(np.arange(self.n_arms), 2)),
                                            names=['var1', 'var2']),
            columns=['n_observations', 'mu1', 'mu2', 'diff_mean', 'z_statistic', 'p_value_zstat', 'se_zstat',
                     'bs_difference_means', 'p_value_bs', 'bs_confident_interval',
                     'winner_z_test', 'winner_bootstrap'])

    def start_experiment(self, seed=1, n_boots=10000):
        np.random.seed(seed)
        data = np.random.binomial(n=[1] * self.n_arms, p=self.p_array_mu, size=(self.n_obs_every_arm, self.n_arms))
        for index, row in self.__all_comparisons_df.iterrows():
            data1, data2 = data[:, index[0]], data[:, index[1]]
            self.__all_comparisons_df.loc[index, "mu1"] = data1.mean()
            self.__all_comparisons_df.loc[index, "mu2"] = data2.mean()
            self.__all_comparisons_df.loc[index, "diff_mean"] = data1.mean() - data2.mean()
            z_stat_ratio, p_value_ztest = proportions_ztest([data1.sum(), data2.sum()],
                                                            [data.shape[0]] * self.n_arms)

            self.__all_comparisons_df.loc[index, "z_statistic"] = z_stat_ratio
            self.__all_comparisons_df.loc[index, "p_value_zstat"] = p_value_ztest
            self.__all_comparisons_df.loc[index, "se_zstat"] = np.sqrt(
                (data1.mean() * (1 - data1.mean()) + data2.mean() * (1 - data2.mean())) / self.n_obs_every_arm)
            data1_bs_sample_means = bootstrap_jit_parallel(data1, n_boots=n_boots)
            data2_bs_sample_means = bootstrap_jit_parallel(data2, n_boots=n_boots)
            difference_bs_means = data1_bs_sample_means - data2_bs_sample_means
            self.__all_comparisons_df.at[index, "bs_difference_means"] = difference_bs_means
            self.__all_comparisons_df.loc[index, "p_value_bs"] = 2 * np.min([np.sum(difference_bs_means < 0) / n_boots,
                                                                             1 - np.sum(difference_bs_means < 0) / n_boots])
        # Sort different ways
        # 1 way - z-test
        self.__all_comparisons_df.sort_values("p_value_zstat", inplace=True)
        self.__all_comparisons_df['i_ztest'] = np.arange(self.__all_comparisons_df.shape[0]) + 1
        self.__all_comparisons_df['alpha_correction_zstat'] = (self.__all_comparisons_df['i_ztest'] * self.alpha) / \
                                                         self.__all_comparisons_df.shape[0]
        self.__all_comparisons_df['stat_significance_z_test'] = np.where(self.__all_comparisons_df['p_value_zstat'] >
                                                                         self.__all_comparisons_df['alpha_correction_zstat'],
                                                                         False, True)
        self.__all_comparisons_df['z_crit_alpha'] = stats.norm.ppf(1 - self.__all_comparisons_df['alpha_correction_zstat'] / 2)
        self.__all_comparisons_df['ci_lower_ztest'] = self.__all_comparisons_df.loc[index, "diff_mean"] - \
                                                      self.__all_comparisons_df['z_crit_alpha'] * self.__all_comparisons_df.loc[index, "se_zstat"]
        self.__all_comparisons_df['ci_upper_ztest'] = self.__all_comparisons_df.loc[index, "diff_mean"] + \
                                                      self.__all_comparisons_df['z_crit_alpha'] * self.__all_comparisons_df.loc[index, "se_zstat"]

        # 2 way - bootstrap
        self.__all_comparisons_df.sort_values(['p_value_bs'], inplace=True)
        self.__all_comparisons_df['i_bootstrap'] = np.arange(self.__all_comparisons_df.shape[0]) + 1
        self.__all_comparisons_df['alpha_correction_bs'] = (self.__all_comparisons_df['i_bootstrap'] * self.alpha) / \
                                                                self.__all_comparisons_df.shape[0]

        # Create confident intervals for difference with correction significance level
        for index, row in self.__all_comparisons_df.iterrows():
            self.__all_comparisons_df.at[index, 'bs_confident_interval'] = get_bs_confidence_interval(
                self.__all_comparisons_df.loc[index, "bs_difference_means"],
                alpha=self.__all_comparisons_df.loc[index, 'alpha_correction_bs'])
        self.__all_comparisons_df["ci_lower_bs"] = self.__all_comparisons_df.loc[:, "bs_confident_interval"].\
            apply(lambda x: x[0])
        self.__all_comparisons_df["ci_upper_bs"] = self.__all_comparisons_df.loc[:, "bs_confident_interval"].\
            apply(lambda x: x[1])

        self.__all_comparisons_df['stat_significance_bs'] = np.where(
            (self.__all_comparisons_df['ci_lower_bs'] > 0) |
            (self.__all_comparisons_df['ci_upper_bs'] < 0), True, False)

        # Determine winners
        for index, row in self.__all_comparisons_df.iterrows():
            self.__all_comparisons_df.loc[index, "winner_z_test"] = np.where(
                (row['stat_significance_z_test'] is True) &
                (row['diff_mean'] < 0), str(index[1]), np.where(
                    row['stat_significance_z_test'] is True, str(index[0]), "not_winner")).item()
        # winner_count_bootstrap = self.__all_comparisons_df['winner'].value_counts()

        for index, row in self.__all_comparisons_df.iterrows():
            self.__all_comparisons_df.loc[index, "winner_bootstrap"] = np.where(
                (row['stat_significance_bs'] is True) &
                (row['diff_mean'] < 0), str(index[1]), np.where(
                    row['stat_significance_bs'] is True, str(index[0]), "not_winner")).item()

        self.__all_comparisons_df['n_observations'] = self.n_obs_every_arm
        winner_df = {"zratio": self.__all_comparisons_df['winner_z_test'].values[0],
                     "bootstrap": self.__all_comparisons_df['winner_bootstrap'].values[0]}
        intermediate_df = self.__all_comparisons_df.copy()
        gc.collect()
        return winner_df, intermediate_df.T


def plot_alpha_power(data: np.ndarray, label: str, ax: Axes,
                     color: str = sns.color_palette("deep")[0],
                     linewidth=3):
    sorted_data = np.sort(data)
    position = stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    return ax.plot(sorted_data,
                   cdf,
                   color=color,
                   linestyle='solid',
                   label=label,
                   linewidth=linewidth)


def create_confidence_plot(df_results, lower, upper):
    for lower, upper, y in zip(df_results[lower],df_results[upper],range(len(df_results))):
        plt.plot((lower,upper),(y,y),'ro-');
    plt.yticks(range(len(df_results)),list(df_results.index));
    plt.axvline(x=0, color='b', ls='--');




