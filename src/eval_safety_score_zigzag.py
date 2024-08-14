import os
import re
import numpy as np
import pandas as pd
from config import *
import glob
import matplotlib.pyplot as plt
from utils import *
from joblib import Parallel, delayed


def load_acc_dataframe(args, eda_feat, tgt_user):
    def load_aggressive(ori_df):
        new_records = []
        window_size = 30  # 30
        cSenID_unique = ori_df['cSenID'].unique()[0]
        cSenDate_unique = ori_df['cSenDate'].unique()[0]
        for i in range(0, len(ori_df) - window_size + 1):
            window = ori_df.iloc[i:i + window_size]
            start_time = window['datelog'].iloc[0]
            end_time = window['datelog'].iloc[-1]
            time_diff = (end_time - start_time).total_seconds()
            accx_diff = window['cSenAccX'].max() - window['cSenAccX'].min()
            if time_diff <= 3.5 and accx_diff <= 40.0:
                new = {
                    'cSenID': cSenID_unique,
                    'cSenDate': cSenDate_unique,
                    'aggressive': accx_diff
                }
                new_records.append(new)
        new_df = pd.DataFrame(new_records)
        return new_df

    def load_zigzag(ori_df):
        window_size = 20
        new_records = []
        cSenID_unique = ori_df['cSenID'].unique()[0]
        cSenDate_unique = ori_df['cSenDate'].unique()[0]
        for i in range(0, len(ori_df) - window_size + 1):
            window = ori_df.iloc[i:i + window_size]
            start_time = window['datelog'].iloc[0]
            end_time = window['datelog'].iloc[-1]
            time_diff = (end_time - start_time).total_seconds()
            angx_diff = window['cSenAngX'].max() - window['cSenAngX'].min()
            if time_diff <= 2.5 and angx_diff <= 40.0:
                new = {
                    'cSenID': cSenID_unique,
                    'cSenDate': cSenDate_unique,
                    'zigzag': angx_diff
                }
                new_records.append(new)
        new_df = pd.DataFrame(new_records)
        return new_df

    def process_file(filename, eda_feat, drop_period):
        df = pd.read_csv(filename)
        if df.shape[0] >= drop_period:
            if eda_feat == 'zigzag':
                df['datelog'] = pd.to_datetime(df['datelog'])
                return load_zigzag(df)
            elif eda_feat == 'aggressive':
                df['datelog'] = pd.to_datetime(df['datelog'])
                return load_aggressive(df)
            else:
                return df[['cSenID', 'cSenDate', eda_feat]]
        return pd.DataFrame()

    file_name = f"{eda_feat}{tgt_user}_.csv"
    dir_path = os.path.join(args.SORT_PATH, str(tgt_user))
    # dir_path = os.path.join(args.SAMPLE_PATH, str(tgt_user))
    result_file_path = os.path.join(dir_path, file_name)

    combined_dfs = []

    if os.path.isdir(dir_path):
        sub_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
        sub_dirs.sort()

        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(dir_path, sub_dir)
            files = glob.glob(os.path.join(sub_dir_path, '*.csv'))
            files.sort()

            eda_dfs = Parallel(n_jobs=30)(
                delayed(process_file)(filename, eda_feat, args.drop_period) for filename in files)
            combined_dfs.extend(eda_dfs)

        combined_df = pd.concat(combined_dfs, ignore_index=True)

        if eda_feat == 'cSenAccX':
            condition = combined_df['cSenID'] == tgt_user
            combined_df.loc[condition, 'cSenAccX'] -= combined_df.loc[condition, 'cSenAccX'].mean()
        elif eda_feat in {'cSenAccY', 'cSenAngX'}:
            condition = combined_df['cSenID'] == int(tgt_user)
            if tgt_user in {'29550344', '92530466'}:
                for date in combined_df[condition]['cSenDate'].unique():
                    cond2 = combined_df['cSenDate'] == date
                    combined_df.loc[condition & cond2, eda_feat] -= combined_df.loc[
                        condition & cond2, eda_feat].mean()
            else:
                combined_df.loc[condition, eda_feat] -= combined_df.loc[condition, eda_feat].mean()

        combined_df.to_csv(result_file_path, index=False)


def safety_interpolation(thre, val):
    def value_interpolation(value):
        if value <= thre[0]:
            return 100
        elif value >= thre[1]:
            return 50
        else:
            # Linear interpolation between 0.1 (100) and 0.3 (70)
            # return 100 - ((value - thre[0]) / (thre[1] - thre[0])) * (100 - 50)

            normalized_value = (value - thre[0]) / (thre[1] - thre[0])
            # Adjust the exponent to control the rate of decay
            exponent = 5
            # nonlinear_interpolation = 50 + (100 - 50) * np.exp(-exponent * normalized_value)
            nonlinear_interpolation = 50 + (100 - 50) * (1 - np.exp(-exponent * (1 - normalized_value)))
            return nonlinear_interpolation

    if isinstance(val, pd.Series):
        result = val.apply(value_interpolation)
    else:
        result = value_interpolation(val)
    return result


if __name__ == "__main__":
    args = set_parameters()
    HEADER_LIST = set_header('202202')

    tmp_df = pd.read_csv(args.SORT_PATH + 'data_distribution_from_202201_to_202407.csv')
    active_user_list = tmp_df.iloc[:, 1:].sum()[tmp_df.iloc[:, 1:].sum() >= args.sample_period].index.tolist()

    tgt_list = ['aggressive', 'zigzag']
    tgt_ratio = 0.4
    tgt_time = 4
    # user_id = '22026979' # '26674039' #'22026979'
    # '22023235', '22052407', '22052437'
    feat_score = {}

    for user_id in tqdm(active_user_list):
        ##### Load dataframe and count #####
        print(f'User ID: {user_id}')

        eda_feat = tgt_list[1]
        if eda_feat == 'aggressive':
            bins = np.arange(0.05, 3.05, 0.05)
        elif eda_feat == 'zigzag':
            bins = np.arange(0.5, 20.5, 0.5)

        result_path = os.path.join(args.EDA_PATH, f"{eda_feat}")
        os.makedirs(result_path, exist_ok=True)

        tgt_file_name = f"{eda_feat}{user_id}_.csv"
        # tgt_file_path = os.path.join(args.SAMPLE_PATH, str(user_id), tgt_file_name)
        tgt_file_path = os.path.join(args.SORT_PATH, str(user_id), tgt_file_name)

        result_file_name = f'{eda_feat}_bin_ratio_{user_id}.csv'
        # result_file_path = os.path.join(args.SAMPLE_PATH, str(user_id), result_file_name)
        result_file_path = os.path.join(args.SORT_PATH, str(user_id), result_file_name)

        # if not os.path.exists(tgt_file_path):
        print(f'[{eda_feat}] Load DataFrame')
        load_acc_dataframe(args, eda_feat, user_id)

        #### count aggressive and zigzag ####
        if os.path.exists(result_file_path):
            cnt_df = pd.read_csv(result_file_path)
        else:
            tmp_df = pd.read_csv(tgt_file_path)
            cnt_df = pd.DataFrame(columns=bins)

            num_batches = len(tmp_df) // args.sample_period

            for i in range(num_batches):
                start_idx = i * args.sample_period
                end_idx = (i + 1) * args.sample_period
                batch_df = tmp_df.iloc[start_idx:end_idx]

                for b in bins:
                    tmp_name = f'{user_id}_{i}'
                    cnt_df.at[tmp_name, b] = round(100 * ((batch_df[eda_feat] > b).sum() / len(batch_df)), 2)

                if len(batch_df) < args.sample_period:
                    continue

                counts, bin_edges = np.histogram(batch_df[eda_feat], bins=300)
                cut_off = 100

                filtered_counts = counts[counts > cut_off]
                filtered_bin_edges = bin_edges[:-1][counts > cut_off]

            pd.DataFrame(cnt_df).to_csv(result_file_path)

        # #### Evaluate Safety Score #####
        # for eda_feat in tgt_list:
        #     if eda_feat == 'aggressive':
        #         bins = np.arange(0.05, 3.05, 0.05)
        #         tgt_bin = str(0.25)
        #         threshold = [10.0, 40.0]
        #     elif eda_feat == 'zigzag':
        #         bins = np.arange(0.5, 20.5, 0.5)
        #         tgt_bin = str(3.0)
        #         threshold = [45.0, 55.0]
        #
        #     score_idx_file = f'{eda_feat}_safety_score_{user_id}.csv'
        #     score_idx_path = os.path.join(args.SORT_PATH, str(user_id), score_idx_file)
        #
        #     cnt_file_name = f'{eda_feat}_bin_ratio_{user_id}.csv'
        #     cnt_file_path = os.path.join(args.SORT_PATH, str(user_id), cnt_file_name)
        #     cnt_df = pd.read_csv(cnt_file_path)
        #     cnt_df.set_index(cnt_df.columns[0], inplace=True)
        #
        #     score_df = cnt_df.rolling(window=args.update_decay, min_periods=1).mean()
        #     if not os.path.exists(score_idx_path):
        #         score_df.to_csv(score_idx_path)





        #
        #         if len(score_df) <= tgt_time:
        #             specific_score = score_df.iloc[-1]
        #         else:
        #             specific_score = score_df.iloc[tgt_time]
        #
        #         specific_score = specific_score[tgt_bin]
        #         specific_score = safety_interpolation(threshold, specific_score)
        #
        #         if eda_feat == 'aggressive':
        #             feat_score['aggressive'] = specific_score
        #         elif eda_feat == 'zigzag':
        #             feat_score['zigzag'] = specific_score
        #
        # final_score = feat_score['aggressive'] * tgt_ratio + feat_score['zigzag'] * (1 - tgt_ratio)
        #
        # print(f'Aggresive Score: {feat_score['aggressive']}')
        # print(f'Zigzag Score: {feat_score['zigzag']}')
        # print(f'Safety Score: {final_score}')