from itertools import combinations
import statsmodels.stats.api as sms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from AB.src.bootstrap import bootstrap_jit_parallel


def get_size_student(mean1, mean2, alpha, beta, sd=None):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    if sd != None:
        n = (np.sqrt(2) * sd * (z_beta + z_alpha) / (mean1 - mean2)) ** 2
    else:
        n = "kek"
    return np.uint16(n)


def get_size_zratio(p1, p2, alpha, beta):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(1 - beta)
    n = (p1 * (1 - p1) + p2 * (1 - p2)) * ((z_alpha + z_beta) / (p1 - p2)) ** 2
    return np.uint16(n)


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
    def __init__(self, data: np.array, alpha=0.05, beta=0.8, equal_var=False):
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

    def student_multiple_test(self) -> tuple[pd.DataFrame, any]:
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
            all_comparisons_student_df.loc[index, "lower"] = \
            cm.tconfint_diff(usevar=uservar, alpha=row['alpha_correction'])[0]
            all_comparisons_student_df.loc[index, "upper"] = \
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

    def mann_whitney_multiple_test(self) -> tuple[pd.DataFrame, any]:
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

    def bootstrap_multiple_test(self, n_boots: int) -> tuple[pd.DataFrame, any]:
        all_comparisons_bootstrap = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(list(combinations(np.arange(self.data.shape[1]), 2)),
                                            names=['var1', 'var2']),
            columns=['bs_difference_means', 'p_value', 'bs_confident_interval'])
        # Calculate bs samples
        for index, row in all_comparisons_bootstrap.iterrows():
            data1_bs_sample_means = bootstrap_jit_parallel(self.data[:, index[0]], n_boots=n_boots)
            data2_bs_sample_means = bootstrap_jit_parallel(self.data[:, index[1]], n_boots=n_boots)
            difference_bs_means = data1_bs_sample_means - data2_bs_sample_means
            all_comparisons_bootstrap.at[index, "bs_difference_means"] = difference_bs_means
            all_comparisons_bootstrap.loc[index, "mean1_mean2_diff"] = np.mean(data1_bs_sample_means) - np.mean(data2_bs_sample_means)
            all_comparisons_bootstrap.loc[index, "p_value"] = 2 * np.min([np.sum(difference_bs_means < 0) / n_boots,
                                                                         1 - np.sum(difference_bs_means < 0) / n_boots])
        all_comparisons_bootstrap.sort_values(['p_value'], inplace=True)
        all_comparisons_bootstrap.loc[:, 'i'] = np.arange(all_comparisons_bootstrap.shape[0]) + 1
        all_comparisons_bootstrap.loc[:, 'alpha_correction'] = (all_comparisons_bootstrap['i'] * self.alpha) / \
                                                                all_comparisons_bootstrap.shape[0]

        # Create confident intervals for difference with correction significance level
        for index, row in all_comparisons_bootstrap.iterrows():
            all_comparisons_bootstrap.at[index, 'bs_confident_interval'] = get_bs_confidence_interval(
                all_comparisons_bootstrap.loc[index, "bs_difference_means"],
                alpha=all_comparisons_bootstrap.loc[index, 'alpha_correction'])
        all_comparisons_bootstrap["lower"] = all_comparisons_bootstrap.loc[:, "bs_confident_interval"].apply(
            lambda x: x[0])
        all_comparisons_bootstrap["upper"] = all_comparisons_bootstrap.loc[:, "bs_confident_interval"].apply(
            lambda x: x[1])

        all_comparisons_bootstrap['stat_significance'] = np.where(
            (all_comparisons_bootstrap['lower'] > 0) |
            (all_comparisons_bootstrap['upper'] < 0), True, False)

        # Determine winners
        for index, row in all_comparisons_bootstrap.iterrows():
            all_comparisons_bootstrap.loc[index, "winner"] = np.where(
                (row['stat_significance'] is True) &
                (row['mean1_mean2_diff'] < 0), str(index[1]), np.where(
                    row['stat_significance'] is True, str(index[0]), "not_winner")).item()
        winner_count_bootstrap = all_comparisons_bootstrap['winner'].value_counts()

        winner = None
        if np.all(np.array(winner_count_bootstrap.index)) == "not_winner":
            winner = "not_winner"

        return all_comparisons_bootstrap, winner_count_bootstrap

