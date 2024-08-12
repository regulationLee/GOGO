import os
import re
import numpy as np
import pandas as pd
from config import *
import glob
import matplotlib.pyplot as plt
from utils import *
from joblib import Parallel, delayed


def load_dataframe(args, eda_feat, tgt_user):
    def extract_number(filename):
        match = re.search(r'_(\d+)\.csv$', filename)
        return int(match.group(1)) if match else float('inf')

    def load_aggressive(ori_df):
        new_records = []
        window_size = 30 # 30
        cSenID_unique = ori_df['cSenID'].unique()[0]
        cSenDate_unique = ori_df['cSenDate'].unique()[0]
        # for i in range(len(ori_df)):
        #     start_idx = max(0, i - window_size + 1)
        #     end_idx = i + 1
        #     window = ori_df.iloc[start_idx:end_idx]
        #     start_time = window['datelog'].iloc[0]
        #     end_time = window['datelog'].iloc[-1]
        #     time_diff = (end_time - start_time).total_seconds()
        #     accx_diff = window['cSenAccX'].max() - window['cSenAccX'].min()
        #     if time_diff <= 2.5 and accx_diff <= 40.0:
        #         new = {
        #             'cSenID': cSenID_unique,
        #             'cSenDate': cSenDate_unique,
        #             'aggressive': accx_diff
        #         }
        #     else:
        #         new = {
        #             'cSenID': cSenID_unique,
        #             'cSenDate': cSenDate_unique,
        #             'aggressive': 0
        #         }
        #     new_records.append(new)
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
        # for i in range(len(ori_df)):
        #     start_idx = max(0, i - window_size + 1)
        #     end_idx = i + 1
        #     window = ori_df.iloc[start_idx:end_idx]
        #     start_time = window['datelog'].iloc[0]
        #     end_time = window['datelog'].iloc[-1]
        #     time_diff = (end_time - start_time).total_seconds()
        #     angx_diff = window['cSenAngX'].max() - window['cSenAngX'].min()
        #     if time_diff <= 2.5 and angx_diff <= 40.0:
        #         new = {
        #             'cSenID': cSenID_unique,
        #             'cSenDate': cSenDate_unique,
        #             'zigzag': angx_diff
        #         }
        #     else:
        #         new = {
        #             'cSenID': cSenID_unique,
        #             'cSenDate': cSenDate_unique,
        #             'zigzag': 0
        #         }
        #     new_records.append(new)
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

    def process_directory(dir_name):
        file_name = f"{eda_feat}{dir_name}_.csv"
        dir_path = os.path.join(args.SAMPLE_PATH, str(dir_name))
        result_file_path = os.path.join(dir_path, file_name)

        if os.path.exists(result_file_path):
            print(f'Load integrated dataframe for {eda_feat}')
            return pd.read_csv(result_file_path)

        combined_dfs = []

        if os.path.isdir(dir_path):
            sub_dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
            sub_dirs.sort()

            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(dir_path, sub_dir)
                files = glob.glob(os.path.join(sub_dir_path, '*.csv'))
                files.sort()

                eda_dfs = Parallel(n_jobs=-1)(
                    delayed(process_file)(filename, eda_feat, args.drop_period) for filename in files)
                combined_dfs.extend(eda_dfs)

            combined_df = pd.concat(combined_dfs, ignore_index=True)

            if eda_feat == 'cSenAccX':
                condition = combined_df['cSenID'] == dir_name
                combined_df.loc[condition, 'cSenAccX'] -= combined_df.loc[condition, 'cSenAccX'].mean()
            elif eda_feat in {'cSenAccY', 'cSenAngX'}:
                condition = combined_df['cSenID'] == int(dir_name)
                if dir_name in {'29550344', '92530466'}:
                    for date in combined_df[condition]['cSenDate'].unique():
                        cond2 = combined_df['cSenDate'] == date
                        combined_df.loc[condition & cond2, eda_feat] -= combined_df.loc[
                            condition & cond2, eda_feat].mean()
                else:
                    combined_df.loc[condition, eda_feat] -= combined_df.loc[condition, eda_feat].mean()

            combined_df.to_csv(result_file_path, index=False)

        return combined_df

    dir_list = sorted([d for d in os.listdir(args.SAMPLE_PATH) if
                       d in tgt_user and os.path.isdir(os.path.join(args.SAMPLE_PATH, d))])

    combined_dfs = Parallel(n_jobs=-1)(delayed(process_directory)(dir_name) for dir_name in dir_list)

    return pd.concat(combined_dfs, ignore_index=True)


if __name__ == "__main__":
    args = set_parameters()
    HEADER_LIST = set_header('202202')
    # tgt_list = ['cSenAccX', HEADER_LIST[5], HEADER_LIST[10], 'aggressive', 'zigzag']
    tgt_list = ['aggressive', 'zigzag']
    user_group_list = [args.accidents, args.non_accidents]

    ##### Load dataframe and count #####
    for eda_feat in tgt_list:
        accumulated_bin_edges = []
        accumulated_bin_count = []

        if eda_feat == 'aggressive':
            bins = np.arange(0.05, 3.05, 0.05)
        elif eda_feat == 'zigzag':
            bins = np.arange(0.5, 20.5, 0.5)

        result_path = os.path.join(args.EDA_PATH, f"{eda_feat}")
        os.makedirs(result_path, exist_ok=True)

        for tgt_user in user_group_list:
            print(f"Load dataframe for {eda_feat}")
            load_dataframe(args, eda_feat, tgt_user)

            for user_id in tgt_user:
                print(f"{eda_feat}_{user_id}")
                tmp_file = f"{eda_feat}{user_id}_.csv"
                tmp_df = pd.read_csv(os.path.join(args.SAMPLE_PATH, str(user_id), tmp_file))

                result_file_name = f'{eda_feat}_bin_ratio_{user_id}.csv'

                if os.path.exists(result_file_name):
                    result_df = pd.read_csv(result_file_name)
                else:
                    result_df = pd.DataFrame(columns=bins)

                num_batches = len(tmp_df) // args.sample_period

                for i in range(num_batches):
                    start_idx = i * args.sample_period
                    end_idx = (i + 1) * args.sample_period
                    batch_df = tmp_df.iloc[start_idx:end_idx]

                    for b in bins:
                        tmp_name = f'{user_id}_{i}'
                        result_df.at[tmp_name, b] = round(100 * ((batch_df[eda_feat] > b).sum() / len(batch_df)), 2)

                    if len(batch_df) < args.sample_period:
                        continue

                    if eda_feat == 'zigzag':
                        counts, bin_edges = np.histogram(batch_df[eda_feat], bins=300)
                        cut_off = 100
                    else:
                        counts, bin_edges = np.histogram(batch_df[eda_feat], bins=300)
                        cut_off = 100

                    filtered_counts = counts[counts > cut_off]
                    filtered_bin_edges = bin_edges[:-1][counts > cut_off]

                    accumulated_bin_edges.append(bin_edges)
                    accumulated_bin_count.append(counts)

                    plt.figure(figsize=(8, 6))
                    plt.bar(filtered_bin_edges, filtered_counts, width=np.diff(bin_edges)[0], edgecolor='black',
                            align='edge')
                    plt.title(f'{eda_feat} {user_id} Batch {i}')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.grid(True)

                    histogram_file_path = f'{eda_feat}_{user_id}_batch_{i}.png'
                    plt.savefig(os.path.join(result_path, histogram_file_path))
                    plt.close()

                preset_bin_ratio_file = os.path.join(args.SAMPLE_PATH, str(user_id), result_file_name)
                pd.DataFrame(result_df).to_csv(preset_bin_ratio_file)

            accumulated_bin_edges_file = os.path.join(result_path, f'{eda_feat}_accidents_bin_edges.csv')
            pd.DataFrame(accumulated_bin_edges).to_csv(accumulated_bin_edges_file, index=False, header=False)

            accumulated_bin_counts_file = os.path.join(result_path, f'{eda_feat}_accidents_bin_counts.csv')
            pd.DataFrame(accumulated_bin_count).to_csv(accumulated_bin_counts_file, index=False, header=False)



    #### Evaluate Safety Score #####
    print(f'Evaluate Safety Score')
    for eda_feat in tgt_list:
        if eda_feat == 'aggressive':
            bins = np.arange(0.05, 3.05, 0.05)
        elif eda_feat == 'zigzag':
            bins = np.arange(0.5, 20.5, 0.5)

        total_score_df = pd.DataFrame()
        total_score_name = f'{eda_feat}_safety_score.csv'
        total_score_path = os.path.join(args.EDA_PATH, eda_feat, total_score_name)

        for tgt_user in user_group_list:

            for user_id in tgt_user:
                score_idx_file = f'{eda_feat}_safety_score_{user_id}.csv'
                score_idx_path = os.path.join(args.SAMPLE_PATH, str(user_id), score_idx_file)

                # if os.path.exists(score_idx_file):
                #     score_df = pd.read_csv(score_idx_file)
                # else:
                #     score_df = pd.DataFrame(columns=bins)

                cnt_file_name = f'{eda_feat}_bin_ratio_{user_id}.csv'
                cnt_file_path = os.path.join(args.SAMPLE_PATH, str(user_id), cnt_file_name)
                cnt_df = pd.read_csv(cnt_file_path)
                cnt_df.set_index(cnt_df.columns[0], inplace=True)

                score_df = cnt_df.rolling(window=args.update_decay, min_periods=1).mean()
                score_df.to_csv(score_idx_path)

                total_score_df = pd.concat([total_score_df, score_df])

        total_score_df.to_csv(total_score_path)


    ##### Evaluate Safety Score #####
    print(f'Compare Safety Score at specific time')
    tgt_time = 4
    final_score = []
    for eda_feat in tgt_list:
        if eda_feat == 'aggressive':
            bins = np.arange(0.05, 3.05, 0.05)
            tgt_bin = str(0.25)
        elif eda_feat == 'zigzag':
            bins = np.arange(0.5, 20.5, 0.5)
            tgt_bin = str(3.0)

        specific_score = []
        specific_score_name = f'{eda_feat}_safety_score_at_{tgt_time}.csv'
        specific_score_path = os.path.join(args.EDA_PATH, eda_feat, specific_score_name)

        for tgt_user in user_group_list:

            for user_id in tgt_user:
                score_idx_file = f'{eda_feat}_safety_score_{user_id}.csv'
                score_idx_path = os.path.join(args.SAMPLE_PATH, str(user_id), score_idx_file)

                tmp_score_df = pd.read_csv(score_idx_path)
                if len(tmp_score_df) <= tgt_time:
                    specific_score.append(tmp_score_df.iloc[-1])
                else:
                    specific_score.append(tmp_score_df.iloc[tgt_time])
                # specific_score_df = pd.concat([specific_score_df, tmp_score_df.iloc[tgt_time, :]], ignore_index=True)

        specific_score_df = pd.DataFrame(specific_score)
        # specific_score_df.to_csv(specific_score_path, index=False)
        specific_score_df.set_index(specific_score_df.columns[0], inplace=True)
        final_score.append(specific_score_df.loc[:, tgt_bin])
    final_score_name = f'final_safety_score_at_{tgt_time}.csv'
    final_score_path = os.path.join(args.EDA_PATH, final_score_name)
    pd.DataFrame(final_score).to_csv(final_score_path)

    #### Test code #####
    final_score_name = f'final_safety_score_at_4.csv'
    final_score_path = os.path.join(args.EDA_PATH, final_score_name)
    tmp_df = pd.read_csv(final_score_path)
    tmp_df = tmp_df.T
    # tmp_df.set_index(tmp_df.columns[0], inplace=True)
    T_score_name = f'T_safety_score_at_4.csv'
    tmp_df.to_csv(os.path.join(args.EDA_PATH, T_score_name))

    ##### Score Tuning #####
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

    T_score_name = f'T_safety_score_at_4.csv'
    tmp_df2 = pd.read_csv(os.path.join(args.EDA_PATH, T_score_name))

    threshold = [10.0, 40.0]
    tmp_df2['agg_inter'] = safety_interpolation(threshold, tmp_df2['aggressive'])
    threshold = [45.0, 55.0]
    tmp_df2['zig_inter'] = safety_interpolation(threshold, tmp_df2['zigzag'])
    tmp_df2['sum'] = tmp_df2['agg_inter'] + tmp_df2['zig_inter']
    ratios = np.arange(0.05, 1.00, 0.05)
    for ratio in ratios:
        new_column_name = f'sum_{ratio:.2f}_{1-ratio:.2f}'
        tmp_df2[new_column_name] = tmp_df2['agg_inter'] * ratio + tmp_df2['zig_inter'] * (1 - ratio)
    tmp_df2.to_csv(os.path.join(args.EDA_PATH, T_score_name), index=False)
