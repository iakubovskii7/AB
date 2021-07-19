import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro
from scipy.stats import jarque_bera
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
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

def param_mean_tests(data1, data2, alpha=0.05):
    """
    :param data1:
    :param data2:
    :param alpha: significance level
    :return: p-value for student
    """
    stat, p = ttest_ind(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Средние равны')
    else:
        print('Средние не равны')
    return p

def non_param_tests(df: pd.DataFrame, alpha=0.05):
    """
    Mann-Whitney and
    :param alpha: significance level for all variants
    :param kwargs: series of data with commas as delimiter
    :return: there were difference or nor
    """
    global stat
    global p
    if df.shape[1] == 2:
        # Use Mann-Whitney
        stat, p = mannwhitneyu(df.iloc[:, 0], df.iloc[:, 1])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print(f'Mann-Whitney test concludes: medians of two samples are IDENTICAL on {alpha} significance')
        else:
            print(f'Mann-Whitney test concludes: medians of two samples are NOT IDENTICAL on {alpha} significance')
    else:
        for col in df.columns:
            locals()['data_' + col] = df[col].dropna().values
        datas = " , ".join('data_' + col for col in df.columns)
        exec(f'''globals()['stat'], globals()['p'] = kruskal({datas})''')
        print('stat=%.3f, p=%.3f' % (globals()['stat'], globals()['p']))
        if globals()['p'] > alpha:
            print(f'Kruskal test concludes: medians of few samples are IDENTICAL on {alpha} significance')
        else:
            print(f'Kruskal test concludes: medians of few samples are NOT IDENTICAL on {alpha} significance')
    stat_final = globals()['stat']
    p_final = globals()['p']
    del stat
    del p
    return stat_final, p_final

# FWER — family-wise error rate

