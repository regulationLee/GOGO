from joblib import Parallel, delayed
from io import StringIO
import os
import glob
from tqdm import tqdm
from config import *
from utils import *
from datetime import datetime, time, timedelta


def add_seconds(t, seconds):
    dt = datetime.combine(datetime.today(), t)
    dt += timedelta(seconds=seconds)
    return dt.time()


def filter_by_difference(values, max_diff):
    if not values:
        return []

    result_value = []
    filtering = True

    while filtering:
        filtered_values = []
        current_value = values[0]
        result_value.append(current_value)
        j = 1
        while j < len(values):
            if values[j] - current_value >= max_diff:
                filtered_values.append(values[j])
            j += 1

        values = filtered_values
        if len(filtered_values) <= 1:
            break

    return result_value


def find_accident(args, tgt_path, save_path):
    tmp_df = pd.read_csv(tgt_path)
    user_id = str(tmp_df['cSenID'].unique()[0])
    tgt_date = str(tmp_df['cSenDate'].unique()[0])
    tmp_df['cSenTime'] = pd.to_datetime(tmp_df['cSenTime'], format='%H:%M:%S').dt.time

    # Filter indices where conditions are met
    tmp_df['GyrX_diff'] = tmp_df['cSenGyrX'].diff().abs()
    tmp_df['GyrY_diff'] = tmp_df['cSenGyrY'].diff().abs()

    gyr_x_indices = tmp_df[tmp_df['GyrX_diff'] > 150].index
    gyr_y_indices = tmp_df[tmp_df['GyrY_diff'] > 80].index
    tot_list = list(set(gyr_y_indices) | set(gyr_x_indices))

    tot_list = filter_by_difference(tot_list, 50)

    file_names = []

    def process_index(idx):
        accident_time = tmp_df.loc[idx, 'cSenTime']
        init_time = add_seconds(accident_time, -10)
        end_time = add_seconds(accident_time, 20)

        tmp_df2 = tmp_df[(tmp_df['cSenTime'] >= accident_time) & (tmp_df['cSenTime'] <= end_time)].copy()

        mean_value_x = tmp_df['cSenAngX'].mean()
        mean_value_y = tmp_df['cSenAngY'].mean()

        tmp_df2['cSenAngX_avg'] = tmp_df2['cSenAngX'] - mean_value_x
        tmp_df2['cSenAngY_avg'] = tmp_df2['cSenAngY'] - mean_value_y

        # count_above_1 = tmp_df2[tmp_df2['cSenAngX'].abs() > 20].shape[0]
        # count_above_2 = tmp_df2[tmp_df2['cSenAngY'].abs() > 50].shape[0]
        count_above_1 = tmp_df2[tmp_df2['cSenAngX_avg'].abs() > 20].shape[0]
        count_above_2 = tmp_df2[tmp_df2['cSenAngY_avg'].abs() > 40].shape[0]
        total_rows = tmp_df2.shape[0]
        if not total_rows == 0:
            percentage_above_1 = (count_above_1 / total_rows) * 100
            percentage_above_2 = (count_above_2 / total_rows) * 100

            if percentage_above_1 >= 65 and percentage_above_2 >= 65:
                result_df = tmp_df[(tmp_df['cSenTime'] >= init_time) & (tmp_df['cSenTime'] <= end_time)]
                hour = accident_time.hour
                minute = accident_time.minute
                second = accident_time.second
                file_name = f"{user_id}_time_{tgt_date}_{hour}_{minute}_{second}.csv"
                file_names.append(file_name[:-4])
                result_path = os.path.join(save_path, file_name)
                result_df.to_csv(result_path, index=False, encoding='utf-8', mode='w')

    Parallel(n_jobs=-1)(delayed(process_index)(idx) for idx in tot_list)

    return file_names


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_file(args, dir_name, path_name, tmp_NAME):
    save_path = os.path.join(args.EDA_PATH, 'accident')
    create_dir_if_not_exists(save_path)
    tgt_path = os.path.join(path_name, tmp_NAME)
    return find_accident(args, tgt_path, save_path)


if __name__ == "__main__":
    args = set_parameters()

    print('Find Pseudo Accident Cases')

    # sample data with velocity
    dir_list = [os.path.join(args.SORT_PATH, d) for d in os.listdir(args.SORT_PATH)
                if os.path.isdir(os.path.join(args.SORT_PATH, d))]
    tasks = []

    for dir_name in dir_list:
        sub_dir_list = [os.path.join(dir_name, d) for d in os.listdir(dir_name)
                        if os.path.isdir(os.path.join(dir_name, d))]
        for sub_dir_name in sub_dir_list:
            files = glob.glob(os.path.join(sub_dir_name, '*.csv'))
            tasks.extend(
                (args, os.path.basename(dir_name), sub_dir_name, os.path.basename(f)) for f in files if os.path.isfile(f))

    results = Parallel(n_jobs=-1)(delayed(process_file)(*task) for task in tqdm(tasks))

    # Combine file names from all results
    all_file_names = [file for sublist in results for file in sublist]
    file_names_df = pd.DataFrame(all_file_names, columns=['file_name'])
    save_path = os.path.join(args.EDA_PATH, 'accident')
    file_names_df.to_csv(os.path.join(save_path, 'accident_time.csv'), index=False)