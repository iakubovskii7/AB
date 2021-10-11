import os
import glob
import pandas as pd
import numpy as np
import joblib
from tqdm.notebook import tqdm
from collections import Counter
from src.ab import get_size_zratio
from p_tqdm import p_map
import matplotlib.pyplot as plt
import plotly.express as px


class TestResultProcessing:
    def __init__(self, folder: str, test_type: str):
        self.path_to_all_files = f"Experiment results/{folder}"
        self.test_type = test_type

    def create_bayes_results(self, filepath) -> pd.DataFrame:
        """
        :param filepath: path for file with experiment results
        :return: 1xn DataFrame with Bayes results
        """

        df = joblib.load(filepath)
        params = filepath.split("/")[-1].split(r"__")
        result_df = pd.DataFrame(data=list(map(lambda x: x.split("=")[1], params))
                          ).T
        result_df.columns = list(map(lambda x: x.split("=")[0], params))
        result_df.set_index(list(map(lambda x: x.split("=")[0], params)), inplace=True)
        result_df['count_winners'] = ''
        result_df['probability_superiority_mean'] = ''
        result_df['probability_superiority_median'] = ''
        result_df['expected_losses'] = ''
        result_df['size'] = ''
        winners = [df[i][0]['probability_superiority'] for i in range(1000)]
        winners = [i if i != "not_winner" else -1 for i in winners]
        result_df.at[result_df.index[0], "count_winners"] = Counter(winners)
        result_df.at[result_df.index[0], 'probability_superiority_mean'] = np.mean(
            [df[i][1]['probability_superiority'][0] for i in range(1000)], axis=0).round(3)
        result_df.at[result_df.index[0], 'probability_superiority_median'] = np.median(
            [df[i][1]['probability_superiority'][0]  for i in range(1000)], axis=0).round(3)
    #     result_df['expected_losses'] = df[i][1]['probability_superiority'][1].round(5)
        result_df['size'] = df[0][1]['probability_superiority'][2]
        return result_df

    def create_thompson_results(self, filepath) -> pd.DataFrame:
        df = joblib.load(filepath)
        params = filepath.split("/")[-1].split(r"__")
        result_df = pd.DataFrame(data=list(map(lambda x: x.split("=")[1], params))
                                 ).T
        result_df.columns = list(map(lambda x: x.split("=")[0], params))
        result_df.set_index(list(map(lambda x: x.split("=")[0], params)), inplace=True)
        result_df['count_winners'] = ''
        result_df.at[result_df.index[0], "count_winners"] = Counter([df[i][0]['probability_superiority']
                                                                     for i in range(1000)])
        # Get share observations out of needed

        p_control = int(result_df.index.levels[1].values[0])
        mde = int(result_df.index.levels[1].values[0])
        result_df['size'] = ''
        result_df.loc[result_df.index[0], 'size'] = get_size_zratio(p_control, mde, 0.05, 0.2)
        shares = tuple(np.cumsum(df[i][1]['probability_superiority'][2], axis=0)[-1] / result_df['size'].values[0]
                       for i in range(1000))
        result_df['share_observations_mean'] = ''
        result_df['share_observations_median'] = ''
        result_df.at[result_df.index[0], 'share_observations_mean'] = np.mean(shares, axis=0).round(3)
        result_df.at[result_df.index[0], 'share_observations_median'] = np.median(shares, axis=0).round(3)

        # Add share correct / incorrect winners
        result_df['share_test_winner'] = result_df['count_winners'] \
            .apply(lambda x: x[1] / 1000)
        result_df['share_not_winner'] = result_df['count_winners'] \
            .apply(lambda x: x[-1] / 1000)
        result_df['share_control_winner'] = result_df['count_winners'] \
            .apply(lambda x: x[0] / 1000)
        return result_df

    def run(self):
        files_list = glob.glob(f"{self.path_to_all_files}/*")
        if self.test_type == "bayesian":
            all_results = p_map(self.create_bayes_results, files_list)
            all_results = pd.concat(all_results)
            return all_results
        if self.test_type == 'thompson':
            all_results = p_map(self.create_thompson_results, files_list)
            all_results = pd.concat(all_results)
            return all_results


test_result = TestResultProcessing(folder="Thompson/Experiment1/",
                                   test_type="thompson")

reslts = test_result.run()
reslts_fp = reslts.query("mde == '0'")

reslts_fp.query("prob_super == '0.8'").reset_index()[['batch_size_share_mu', 'share_not_winner']]

reslts_fn = reslts.query("mde != '0'")

reslts_fn.query("prob_super == '0.95'").reset_index()[
    ['batch_size_share_mu', 'share_test_winner']
].astype(float).corr()

