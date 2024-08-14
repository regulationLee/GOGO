import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin, isnan
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

import glob
import json


def eval_stat(path, period=None):
    def process_file(file_path, dir_name):
        tmp_df = pd.read_csv(file_path)
        date = tmp_df['cSenDate'].unique()[0]
        return date, dir_name, tmp_df.shape[0]

    dir_list = os.listdir(path)
    dir_list = [f for f in dir_list if f != '.DS_Store' and os.path.isdir(os.path.join(path, f))]
    dir_list.sort()

    columns = ['Date'] + dir_list
    stat_df = pd.DataFrame(columns=columns)
    stat_df.set_index('Date', inplace=True)

    pd.set_option('future.no_silent_downcasting', True)

    ####### Parallel

    if period:
        print('Data Statistic Evaluation: Sorted {}'.format(str(period)))
        result_file = 'data_distribution_' + str(period) + '.csv'
        result_path = os.path.join(path, result_file)

        tasks = []
        for dir_name in dir_list:
            path_name = os.path.join(path, dir_name, str(period))
            files = glob.glob(os.path.join(path_name, '*.csv'))
            if files:
                files.sort()
                tasks.extend((file_path, dir_name) for file_path in files)

    else:
        print('Data Statistic Evaluation')
        result_file = f'data_distribution_from_202201_to_202407.csv'
        result_path = os.path.join(path, result_file)

        if os.path.exists(result_path):
            return

        tasks = []
        for dir_name in dir_list:
            sub_dir_list = os.listdir(path + dir_name)
            sub_dir_list = [f for f in sub_dir_list if f != '.DS_Store' and os.path.isdir(os.path.join(path + dir_name, f))]
            sub_dir_list.sort()
            for sub_dir_name in sub_dir_list:
                path_name = os.path.join(path, dir_name, sub_dir_name)
                files = glob.glob(os.path.join(path_name, '*.csv'))
                if files:
                    files.sort()
                    tasks.extend((file_path, dir_name) for file_path in files)


    results = Parallel(n_jobs=-1)(delayed(process_file)(file_path, dir_name) for file_path, dir_name in tqdm(tasks))

    for date, dir_name, count in results:
        stat_df.at[date, dir_name] = count
    ##########################


    # # ####### if single #######
    # if period:
    #     print('Data Statistic Evaluation: Sorted {}'.format(str(period)))
    #     result_file = 'data_distribution_' + str(period) + '.csv'
    #     result_path = os.path.join(path, result_file)
    #
    #     for dir_name in tqdm(dir_list):
    #         path_name = os.path.join(path, dir_name)
    #         path_name = os.path.join(path_name, str(period))
    #         files = glob.glob(os.path.join(path_name, '*.csv'))
    #         if len(files) != 0:
    #             files.sort()
    #             for file_path in files:
    #                 print(file_path)
    #                 tmp_df = pd.read_csv(file_path)
    #                 date = tmp_df['cSenDate'].unique()[0]
    #                 stat_df.at[date, dir_name] = tmp_df.shape[0]
    #
    # else:
    #     print('Data Statistic Evaluation')
    #     result_file = f'data_distribution_from_2022_to_202403.csv'
    #     result_path = os.path.join(path, result_file)
    #
    #     if os.path.exists(result_path):
    #         return
    #
    #     for dir_name in tqdm(dir_list):
    #         sub_dir_list = os.listdir(path + dir_name)
    #         sub_dir_list = [f for f in sub_dir_list if f != '.DS_Store' and os.path.isdir(os.path.join(path + dir_name, f))]
    #         sub_dir_list.sort()
    #         for sub_dir_name in sub_dir_list:
    #             path_name = os.path.join(path, dir_name, sub_dir_name)
    #             files = glob.glob(os.path.join(path_name, '*.csv'))
    #             if len(files) != 0:
    #                 files.sort()
    #                 for file_path in files:
    #                     print(file_path)
    #                     tmp_df = pd.read_csv(file_path)
    #                     date = tmp_df['cSenDate'].unique()[0]
    #                     stat_df.at[date, dir_name] = tmp_df.shape[0]



    # for dir_name in tqdm(dir_list):
    #     path_name = os.path.join(path, dir_name)
    #     path_name = os.path.join(path_name, str(period))
    #     files = glob.glob(os.path.join(path_name, '*.csv'))
    #     if len(files) != 0:
    #         files.sort()
    #         for file_path in files:
    #             print(file_path)
    #             tmp_df = pd.read_csv(file_path)
    #             date = tmp_df['cSenDate'].unique()[0]
    #             stat_df.at[date, dir_name] = tmp_df.shape[0]
    # #########################

    stat_df.fillna(0, inplace=True)
    stat_df = stat_df.astype(float)
    stat_df = stat_df.sort_index(ascending=True)
    stat_df.to_csv(result_path)

    # sns.heatmap(stat_df, annot=True, cmap='viridis')
    # plt.title('Data Distribution')
    # plt.xlabel('Columns')
    # plt.ylabel('Index')
    # plt.show()
    # plt.close()

def haversine_3d(lat1, lon1, alt1, lat2, lon2, alt2):
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dalt = alt2 - alt1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    distance = sqrt(c ** 2 + (dalt / 1000) ** 2) * R

    return distance

def find_ranges(indices):
    ranges = []
    start = None
    count = 0

    for i in range(len(indices)):
        if start is None:
            start = indices[i]
            count = 1
        elif indices[i] == indices[i - 1] + 1:
            count += 1
        else:
            ranges.append((start, count))
            start = indices[i]
            count = 1

    if start is not None:
        ranges.append((start, count))

    return ranges


def sample_data(args, file_path, result_path):
    # if os.path.exists(result_path):
    #     print('{}: Target data already sampled'.format(result_path))
    #     return

    print('{}: Data sampling'.format(result_path))
    test_df = pd.read_csv(file_path)
    test_df['datelog'] = pd.to_datetime(test_df['datelog'])
    test_df['cSenTime'] = pd.to_datetime(test_df['cSenTime'], format='%H:%M:%S')

    # melted_gps_df = test_df[['gpsLatitude', 'gpsLongitude', 'gpsAltitude']]
    # melted_gps_df = melted_gps_df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
    # unique_gps = melted_gps_df.unique()
    # unique_gps_indices = []
    # for val in tqdm(unique_gps):
    #     idx = melted_gps_df[melted_gps_df == val].index[0]
    #     if idx not in unique_gps_indices:
    #         unique_gps_indices.append(idx)
    #
    # test_df.loc[unique_gps_indices, 'lat_diff'] = test_df.loc[unique_gps_indices, 'gpsLatitude'].diff()
    # test_df.loc[unique_gps_indices, 'lon_diff'] = test_df.loc[unique_gps_indices, 'gpsLongitude'].diff()
    # test_df.loc[unique_gps_indices, 'alt_diff'] = test_df.loc[unique_gps_indices, 'gpsAltitude'].diff()

    melted_gps_df = test_df[['gpsLatitude', 'gpsLongitude', 'gpsAltitude']].astype(str).agg(', '.join, axis=1)
    unique_gps_indices = melted_gps_df.drop_duplicates().index

    for col in ['gpsLatitude', 'gpsLongitude', 'gpsAltitude']:
        test_df[f'{col}_diff'] = test_df[col].diff()

    earth_radius_km = 6371.0
    lat_diff = test_df.loc[unique_gps_indices, 'gpsLatitude_diff'] * (np.pi / 180) * earth_radius_km * 1000
    lon_diff = test_df.loc[unique_gps_indices, 'gpsLongitude_diff'] * (np.pi / 180) * earth_radius_km * 1000 * np.cos(
        test_df.loc[unique_gps_indices, 'gpsLatitude'] * np.pi / 180)

    test_df.loc[unique_gps_indices, 'horizontal_distance'] = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
    test_df.loc[unique_gps_indices, 'total_distance'] = np.sqrt(
        test_df.loc[unique_gps_indices, 'horizontal_distance'] ** 2 + test_df.loc[
            unique_gps_indices, 'gpsAltitude_diff'] ** 2)
    test_df.loc[0, 'total_distance'] = 0

    # test_df.loc[unique_gps_indices, 'horizontal_distance'] = np.sqrt(
    #     (test_df.loc[unique_gps_indices, 'lat_diff'] * (np.pi / 180) * earth_radius_km * 1000) ** 2 +
    #     (test_df.loc[unique_gps_indices, 'lon_diff'] * (np.pi / 180)
    #      * earth_radius_km * 1000 * np.cos(test_df.loc[unique_gps_indices, 'gpsLatitude'] * np.pi / 180)) ** 2)
    #
    # test_df.loc[unique_gps_indices, 'total_distance'] = \
    #     np.sqrt(test_df.loc[unique_gps_indices, 'horizontal_distance'] ** 2 + test_df.loc[
    #         unique_gps_indices, 'alt_diff'] ** 2)
    # test_df.loc[0, 'total_distance'] = 0
    # non_zero_idx = test_df[test_df['total_distance'] != 0].index.tolist()

    tmp_df = test_df.loc[unique_gps_indices, 'datelog'].diff().dt.total_seconds().replace(0, 1)
    test_df.loc[unique_gps_indices, 'time_diff'] = tmp_df

    test_df.loc[unique_gps_indices, 'gpsSpeed_sakak'] = \
        (test_df.loc[unique_gps_indices, 'total_distance'] / test_df.loc[unique_gps_indices, 'time_diff']) * 3.6

    test_df = test_df.fillna(0)
    zero_indices = test_df[test_df['gpsSpeed'] <= args.stop_threshold].index

    ranges = find_ranges(zero_indices)

    filtered_df = test_df.drop(
        columns=['cSenTemp', 'gpsLatitude_diff', 'gpsLongitude_diff', 'gpsAltitude_diff', 'horizontal_distance'])
    # filtered_df = test_df.drop(columns=['cSenTemp', 'lat_diff', 'lon_diff', 'alt_diff', 'horizontal_distance'])
    if filtered_df.columns[0] == 'Unnamed: 0':
        filtered_df = filtered_df.drop(filtered_df.columns[0], axis=1)

    for start, length in ranges:
        if length >= args.stop_period:
            filtered_df = filtered_df.drop(range(start, start + length))

    # file_name = str(filtered_df['cSenID'].iloc[0]) + '_' + str(filtered_df['cSenDate'].iloc[0]) + '.csv'
    if filtered_df.size != 0:
        filtered_df.to_csv(result_path, index=False)

    # file_name_pre = str(filtered_df['cSenID'].iloc[0]) + '_' + str(filtered_df['cSenDate'].iloc[0])
    #
    # for i in range(int(filtered_df.shape[0] / args.sample_period)):
    #     file_name = file_name_pre + '_' + str(i) + '.csv'
    #     if filtered_df.shape[0] <= args.sample_period:
    #         tmp_df = filtered_df
    #     else:
    #         tmp_df = filtered_df.iloc[(i * args.sample_period):(i * args.sample_period + args.sample_period)]
    #     tmp_df.to_csv(os.path.join(result_path, file_name), index='False')


def tgt_visualization(args, tgt_file, feature, threshold):
    tgt_df = pd.read_csv(tgt_file)

    counts, bin_edges = np.histogram(tgt_df[feature], bins=1000)
    cut_off = 100

    filtered_counts = counts[counts > cut_off]
    filtered_bin_edges = bin_edges[:-1][counts > cut_off]

    plt.figure(figsize=(10, 5))  # Set the size of the plot
    plt.bar(filtered_bin_edges, filtered_counts, width=np.diff(bin_edges)[0], edgecolor='black', align='edge')

    plt.axvline(x=threshold[0], color='red', linestyle='-', linewidth=2)
    plt.axvline(x=threshold[1], color='red', linestyle='-', linewidth=2)

    # Show plot
    plt.grid(True)  # Adding grid for better readability
    plt.show()
    plt.close()
    return

def interpolation(args, val):
    def value_interpolation(value):
        if value <= args.index_threshold[0]:
            return 100
        elif value >= args.index_threshold[1]:
            return 50
        else:
            # Linear interpolation between 0.1 (100) and 0.3 (70)
            return 100 - ((value - args.index_threshold[0]) /
                          (args.index_threshold[1] - args.index_threshold[0])) * (100 - 70)
    if isinstance(val, pd.Series):
        result = val.apply(value_interpolation)
    else:
        result = value_interpolation(val)
    return result

def decayed_sum(args, values):
    window_size = args.update_decay

    # simple averaging through sliding window
    result = values.rolling(window=window_size, min_periods=1).mean()

    # when decaying is needed
    # result = []
    # for i in range(len(values)):
    #     if i < window_size - 1:
    #         # For the first few elements, use equal weights (no decay)
    #         decay_factors = np.ones(i + 1) / (i + 1)
    #     else:
    #         # Calculate decay factor for the window
    #         decay_factors = np.exp(-np.arange(window_size))
    #         decay_factors /= np.sum(decay_factors)  # Normalize to make sum equal to 1
    #
    #     # Calculate decayed sum for the window
    #     start_index = max(0, i - window_size + 1)
    #     sum_value = np.sum(values[start_index:i + 1] * decay_factors[:i - start_index + 1])
    #     result.append(sum_value)
    return result


def visualization(args, file_name):
    pd.set_option('future.no_silent_downcasting', True)

    with open(os.path.join(args.RESULT_PATH, file_name), 'rb') as file:
        cnt_result_dict = pickle.load(file)

    # dates = pd.date_range(start='2022-02-01', end='2022-02-28')
    # formatted_dates = [date.strftime('%Y.%m.%d') for date in dates]
    id_values = list(cnt_result_dict.keys())
    columns = ['Date'] + id_values
    score_df = pd.DataFrame(columns=columns)
    score_sum_df = pd.DataFrame(columns=id_values)
    # score_df['Date'] = formatted_dates
    score_df.set_index('Date', inplace=True)

    print('Average Score for each User')

    for key, count_df in cnt_result_dict.items():
        def compute_score(tmp_df, idx):
            #
            # sliding window
            #
            num1 = tmp_df[tmp_df['Date'] == idx]['1st_Abnormals'].sum()
            num2 = tmp_df[tmp_df['Date'] == idx]['2nd_Abnormals'].sum()
            # den = tmp_df[tmp_df['Date'] == idx]['Total'].sum()
            if num1 == 0 or isnan(num1):
                tmp_score = 100
            else:
                tmp_score = interpolation(args, num2/num1)
            return round(tmp_score, 2)

        date_list = count_df['Date'].unique()
        date_list.sort()
        default_val = 0
        # for date in score_df.index.tolist():
        #     if date in data_list:
        #         # default_val = compute_score(count_df, date)
        #         score_df.at[date, key] = compute_score(count_df, date)
        #     else:
        #         score_df.at[date, key] = default_val

        for date in date_list:
            score_df.at[date, key] = compute_score(count_df, date)

        filtered_score_list = [value for value in score_df[str(key)].tolist() if value != 0]
        filtered_score_list = [value for value in filtered_score_list if not isnan(value)]
        avg_score = round(sum(filtered_score_list)/len(filtered_score_list), 1)
        score_sum_df.at['Time_h', key] = round(count_df['Total'].sum()/36000, 2)
        score_sum_df.at['Avg', key] = avg_score
        if count_df['1st_Abnormals'].sum() == 0:
            batch_score = 100
        else:
            batch_score = round(interpolation(args, count_df['2nd_Abnormals'].sum() / count_df['1st_Abnormals'].sum()))
        score_sum_df.at['Batch', key] = batch_score
        print('{}_{} times: Batch {}, Average {}'.format(key, len(filtered_score_list), batch_score, avg_score))

    score_df.fillna(0, inplace=True)
    score_df = score_df.astype(float)
    score_df = score_df.sort_index(ascending=True)
    score_df = pd.concat([score_sum_df, score_df])

    result_file_name = file_name[:-4] + '.csv'
    score_df.to_csv(os.path.join(args.RESULT_PATH, result_file_name))

    categories = score_df.index.to_list()
    values1 = np.zeros(len(categories))
    values2 = np.zeros(len(categories))
    values3 = np.zeros(len(categories))
    for i in range(len(categories)):
        values = score_df.loc[categories[i]].values
        values = [item for item in values if item != 0]
        counts, bin_edges = np.histogram(values, bins=np.arange(70, 110, 10))
        # values1[i] = 100 * (counts[0] / sum(counts))
        # values2[i] = 100 * (counts[1] / sum(counts))
        # values3[i] = 100 * (counts[2] / sum(counts))
        values1[i] = int(counts[0])
        values2[i] = int(counts[1])
        values3[i] = int(counts[2])

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(categories, values1, label='70 ~ 80')
    bars2 = plt.bar(categories, values2, bottom=values1, label='80 ~ 90')
    bars3 = plt.bar(categories, values3, bottom=values1 + values2, label='90 ~ 100')

    for bar1, bar2, bar3 in zip(bars1, bars2, bars3):
        height1 = round(bar1.get_height(), 1)
        height2 = round(bar2.get_height(), 1)
        height3 = round(bar3.get_height(), 1)
        plt.text(bar1.get_x() + bar1.get_width() / 2., height1 / 2, f'{height1}', ha='center', va='bottom', fontsize=10,
                 color='black')
        plt.text(bar2.get_x() + bar2.get_width() / 2., height1 + height2 / 2, f'{height2}', ha='center', va='bottom',
                 fontsize=10, color='black')
        plt.text(bar3.get_x() + bar3.get_width() / 2., height1 + height2 + height3 / 2, f'{height3}', ha='center', va='bottom',
                 fontsize=10, color='black')

    plt.title('Proportion of the Safety Index')
    plt.xlabel('Dates')
    plt.ylabel('User Percentage')
    plt.legend()

    plt.xticks(rotation=45)

    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for column in score_df.columns:
        if str(column) in args.accidents:
            plt.plot(score_df.index, score_df[column], marker='o', markersize=8, markeredgewidth=2, linestyle='', label=column)
        else:
            plt.plot(score_df.index, score_df[column], color='black', marker='x', markeredgewidth=2, linestyle='')

    plt.title(file_name[:-4])
    plt.xlabel('Date')
    plt.ylabel('Safety Index')
    # plt.legend()

    plt.ylim(69, 101)
    plt.xticks(rotation=45)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.close()


def visualization_report(args, file_name):
    pd.set_option('future.no_silent_downcasting', True)

    with open(os.path.join(args.RESULT_PATH, file_name), 'rb') as file:
        cnt_result_dict = pickle.load(file)

    id_values = list(cnt_result_dict.keys())
    score_df = pd.DataFrame(columns=id_values)
    # score_df.set_index('Date', inplace=True)

    # fig, ax = plt.subplots()
    # positions = range(1, len(cnt_result_dict) + 1)

    # for key, count_df in cnt_result_dict.items():
    #     # tmp_score = count_df['Decayed_Score']
    #     # score_df.loc[:, key] = tmp_score
    #     # print('debug')
    #     ax.boxplot(count_df['Decayed_Score'], positions=positions, patch_artist=True,
    #                boxprops=dict(facecolor='skyblue'))

    plt.figure(figsize=(8, 6))
    positions = range(1, len(cnt_result_dict) + 1)
    plt.boxplot([cnt_result_dict[key]['Decayed_Score'] for key in cnt_result_dict],
                positions=positions, labels=list(cnt_result_dict.keys()))

    plt.xlabel('DataFrames')
    plt.ylabel('Values')
    plt.title('Boxplot of Values in Multiple DataFrames')

    plt.tight_layout()
    plt.show()
    plt.close()

