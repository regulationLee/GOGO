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

    interpolated_score_df = pd.DataFrame()

    for user_id in tqdm(active_user_list):
        ##### Load dataframe and count #####
        print(f'User ID: {user_id}')

        #### Evaluate Safety Score #####
        for eda_feat in tgt_list:
            if eda_feat == 'aggressive':
                bins = np.arange(0.05, 3.05, 0.05)
                tgt_bin = str(0.25)
                threshold = [10.0, 40.0]
            elif eda_feat == 'zigzag':
                bins = np.arange(0.5, 20.5, 0.5)
                tgt_bin = str(3.0)
                threshold = [45.0, 55.0]

            score_idx_file = f'{eda_feat}_safety_score_{user_id}.csv'
            score_idx_path = os.path.join(args.SORT_PATH, str(user_id), score_idx_file)

            cnt_file_name = f'{eda_feat}_bin_ratio_{user_id}.csv'
            cnt_file_path = os.path.join(args.SORT_PATH, str(user_id), cnt_file_name)
            cnt_df = pd.read_csv(cnt_file_path)
            cnt_df.set_index(cnt_df.columns[0], inplace=True)

            score_df = cnt_df.rolling(window=args.update_decay, min_periods=1).mean()
            if not os.path.exists(score_idx_path):
                score_df.to_csv(score_idx_path)

            if eda_feat == 'aggressive':
                tmp_agg_score = score_df[tgt_bin].apply(lambda x: safety_interpolation(threshold, x))
                # aggressive_df = pd.concat([aggressive_df, tmp_agg_score], axis=0)
            elif eda_feat == 'zigzag':
                tmp_zig_score = score_df[tgt_bin].apply(lambda x: safety_interpolation(threshold, x))
                # zigzag_df = pd.concat([zigzag_df, tmp_zig_score], axis=0)

        if len(tmp_agg_score) == len(tmp_zig_score):
            tmp_int_score = pd.DataFrame({'aggressive': tmp_agg_score, 'zigzag': tmp_zig_score})
        else:
            if len(tmp_agg_score) == 0 or len(tmp_zig_score) == 0:
                pass
            else:
                diff_num = abs(len(tmp_agg_score) - len(tmp_zig_score))
                if len(tmp_agg_score) > len(tmp_zig_score):
                    last_value = tmp_zig_score.iloc[-1]
                    add_value = pd.Series([last_value] * diff_num)
                    tmp_zig_score = pd.concat([tmp_zig_score, add_value], ignore_index=False)
                    tmp_zig_score.index = tmp_agg_score.index
                else:
                    last_value = tmp_agg_score.iloc[-1]
                    add_value = pd.Series([last_value] * diff_num)
                    tmp_agg_score = pd.concat([tmp_agg_score, add_value], ignore_index=False)
                    tmp_agg_score.index = tmp_zig_score.index
                tmp_int_score = pd.DataFrame({'aggressive': tmp_agg_score, 'zigzag': tmp_zig_score})

        interpolated_score_df = pd.concat([interpolated_score_df, tmp_int_score], axis=0)

    interpolated_score_df['final_score'] = (
            interpolated_score_df['aggressive'] * tgt_ratio + interpolated_score_df['zigzag'] * (1 - tgt_ratio))

    sort_result = interpolated_score_df.sort_values(by='final_score', ascending=False)

    plt.figure(figsize=(10,6))
    plt.plot(sort_result['final_score'], marker='o')
    plt.grid(True)
    plt.show()


                # if len(score_df) <= tgt_time:
                #     specific_score = score_df.iloc[-1]
                # else:
                #     specific_score = score_df.iloc[tgt_time]
                #
                # specific_score = specific_score[tgt_bin]
                # specific_score = safety_interpolation(threshold, specific_score)
                #
                # if eda_feat == 'aggressive':
                #     feat_score['aggressive'] = specific_score
                # elif eda_feat == 'zigzag':
                #     feat_score['zigzag'] = specific_score
        #
        # final_score = feat_score['aggressive'] * tgt_ratio + feat_score['zigzag'] * (1 - tgt_ratio)
        #
        # print(f'Aggresive Score: {feat_score['aggressive']}')
        # print(f'Zigzag Score: {feat_score['zigzag']}')
        # print(f'Safety Score: {final_score}')