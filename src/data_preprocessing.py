from joblib import Parallel, delayed
from io import StringIO
from datetime import datetime
import os
import glob
import pandas as pd
from tqdm import tqdm
from config import *
from utils import *


def sort_rawdata(args, file_path_list):

    def process_file(file_path, args):
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        csv_string = binary_data.decode('utf-8').replace('\x00', '')

        na_values = ['\\N', r'\N']
        df = pd.read_csv(StringIO(csv_string), header=None, na_values=na_values, keep_default_na=True)
        # df = (pd.read_csv(StringIO(csv_string), na_values=na_values, keep_default_na=True).
        #       dropna(subset=['cSenID']))

        if len(df) == 0:
            pass

        if df.shape[1] == 21:
            df.columns = set_header('202111')
        elif df.shape[1] == 22:
            df.columns = set_header('202202')
        elif df.shape[1] == 23:
            df.columns = set_header('202212')
        else:
            df.columns = set_header('202402')
        # df.columns = args.HEADER_LIST
        df = df.dropna(subset=['cSenID'])
        file_date = file_path[-12:-4]
        file_date = datetime.strptime(file_date, '%Y%m%d').strftime('%Y-%m-%d')
        df['cSenDate'] = file_date
        # df['cSenDate'] = df['cSenDate'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d').strftime('%Y-%m-%d')
        #                                       if '-' not in str(x) else x)
        df['cSenID'] = df['cSenID'].astype(int).astype(str).str[-8:].astype(int)
        unique_num_id = df['cSenID'].unique()

        for tmp_val in unique_num_id:
            tmp_id_df = df.loc[df['cSenID'] == tmp_val]
            save_path = os.path.join(args.SORT_PATH, str(tmp_val))
            os.makedirs(save_path, exist_ok=True)

            period = file_path.split('/')[-1].split('_')[-1][0:6]

            save_path = os.path.join(save_path, str(period))
            os.makedirs(save_path, exist_ok=True)

            save_file_prefix = str(tmp_val)

            if len(tmp_id_df['cSenDate'].unique()) != 1:
                filtered_tmp_id_df = tmp_id_df[tmp_id_df['cSenDate'] == file_date]
                save_file = save_file_prefix + '_' + str(file_date) + "_.csv"
                if os.path.exists(os.path.join(save_path, save_file)):
                    print('[Already Sorted] {}'.format(save_file))
                else:
                    filtered_tmp_id_df.to_csv(os.path.join(save_path, save_file), index=False)
            else:
                save_file = save_file_prefix + '_' + str(tmp_id_df['cSenDate'].unique()[0]) + "_.csv"
                if os.path.exists(os.path.join(save_path, save_file)):
                    print('[Already Sorted] {}'.format(save_file))
                else:
                    tmp_id_df.to_csv(os.path.join(save_path, save_file), index=False)

            # save_file = save_file_prefix + '_' + str(tmp_id_df['cSenDate'].unique()[0]) + "_.csv"
            # save_file_path = os.path.join(save_path, save_file)
            #
            # # if os.path.exists(save_file_path):
            # #     print(f'[Already Sorted] {save_file}')
            # # else:
            # tmp_id_df.to_csv(save_file_path, index=False)

    print('Sort data by ID')

    ### if parallel
    Parallel(n_jobs=-1)(
        delayed(process_file)(file_path, args) for file_path in tqdm(file_path_list)
    )


    # ### if single
    # # file_name_list = [os.path.basename(f) for f in files_raw if os.path.isfile(f)]
    # for file_path in tqdm(file_path_list):
    #     file_path = os.path.join(args.RAW_PATH, 'tb_sensordata_20220608.csv')
    #
    #     with open(file_path, 'rb') as f:
    #         binary_data = f.read()
    #     csv_string = binary_data.decode('utf-8').replace('\x00', '')
    #
    #     na_values = ['\\N', r'\N']
    #     df = pd.read_csv(StringIO(csv_string), header=None, na_values=na_values, keep_default_na=True)
    #
    #     # df = pd.read_csv(StringIO(csv_string), header=None, na_values=na_values, keep_default_na=True)
    #     if len(df) == 0:
    #         pass
    #
    #     print(file_path)
    #     if df.shape[1] == 21:
    #         df.columns = set_header('202111')
    #     elif df.shape[1] == 22:
    #         df.columns = set_header('202202')
    #     elif df.shape[1] == 23:
    #         df.columns = set_header('202212')
    #     else:
    #         df.columns = set_header('202402')
    #     # df.columns = args.HEADER_LIST
    #     # df = pd.read_csv(file_path)
    #     df = df.dropna(subset=['cSenID'])
    #     df['cSenDate'] = df['cSenDate'].apply(
    #         lambda x: pd.to_datetime(x, format='%Y%m%d').strftime('%Y-%m-%d') if '-' not in str(x) else x)
    #     df['cSenID'] = df['cSenID'].astype(int).astype(str).str[-8:].astype(int)
    #     # df['cSenID'] = df['cSenID'].astype(str).str[-8:].astype(int)
    #     unique_num_id = df['cSenID'].unique()
    #     for tmp_val in unique_num_id:
    #         tmp_id_df = df.loc[df['cSenID'] == tmp_val]
    #         save_path = args.SORT_PATH + str(tmp_val) + '/'
    #         if not os.path.exists(args.SORT_PATH + str(tmp_val)):
    #             os.mkdir(save_path)
    #
    #         period = file_path.split('/')[-1].split('_')[-1][0:6]
    #
    #         save_path = os.path.join(save_path, str(period))
    #         if not os.path.exists(save_path):
    #             os.mkdir(save_path)
    #
    #         save_file_prefix = str(tmp_val)
    #         if len(tmp_id_df['cSenDate'].unique()) != 1:
    #             file_date = file_path[-12:-4]
    #             file_date = datetime.strptime(file_date, '%Y%m%d').strftime('%Y-%m-%d')
    #             filtered_tmp_id_df = tmp_id_df[tmp_id_df['cSenDate'] == file_date]
    #             save_file = save_file_prefix + '_' + str(file_date) + "_.csv"
    #             if os.path.exists(os.path.join(save_path, save_file)):
    #                 print('[Already Sorted] {}'.format(save_file))
    #             else:
    #                 filtered_tmp_id_df.to_csv(os.path.join(save_path, save_file), index=False)
    #         else:
    #             save_file = save_file_prefix + '_' + str(tmp_id_df['cSenDate'].unique()[0]) + "_.csv"
    #             if os.path.exists(os.path.join(save_path, save_file)):
    #                 print('[Already Sorted] {}'.format(save_file))
    #             else:
    #                 tmp_id_df.to_csv(os.path.join(save_path, save_file), index=False)

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_file(args, dir_name, path_name, target_period, tmp_NAME):
    save_path = os.path.join(args.SAMPLE_PATH, dir_name, str(target_period))
    create_dir_if_not_exists(save_path)
    save_path = os.path.join(save_path, tmp_NAME)
    tgt_path = os.path.join(path_name, tmp_NAME)
    sample_data(args, tgt_path, save_path)


def get_file_name_list(args):
    for filename in os.listdir(args.RAW_PATH):
        # Check if the file matches the pattern
        if filename.startswith('tb_sensordata_') and filename.endswith('.csv'):
            # Split the filename to remove the unwanted part
            parts = filename.split('_')
            if len(parts) == 4:
                new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}.csv"
                # Create full path for old and new filenames
                old_file = os.path.join(args.RAW_PATH, filename)
                new_file = os.path.join(args.RAW_PATH, new_filename)
                # Rename the file
                os.rename(old_file, new_file)

    files_raw = glob.glob(os.path.join(args.RAW_PATH, '*.csv'))
    files_raw.sort()

    return files_raw


if __name__ == "__main__":
    args = set_parameters()

    # target_period = args.COLLECT_PERIOD[2]
    target_index = ['202406', '202407']
    period_list = [i for i in target_index]
    print('Data Preprocessing for {}'.format(period_list))

    file_name_list = get_file_name_list(args)

    sort_rawdata(args, file_name_list)
    eval_stat(args.SORT_PATH)

    tmp_df = pd.read_csv(args.SORT_PATH + 'data_distribution_from_202111_to_202407.csv')
    active_user_list = tmp_df.iloc[:, 1:].sum()[tmp_df.iloc[:, 1:].sum() >= args.sample_period].index.tolist()



    # for target_period in period_list:
    #     print('Period: {}'.format(target_period))
    #
    #     # set folder of rawdata files name as 'raw_202202'
    #     DATA_PATH = 'data/' + 'raw_' + target_period + '/'
    #
    #     ####### Filtered by USER ID #######
    #     sort_rawdata(args, DATA_PATH, target_period)
    #     eval_stat(args.SORT_PATH, target_period)
    #
    #     #### sample data with velocity
    #     dir_list = os.listdir(args.SORT_PATH)
    #     dir_list.sort()
    #
    #     ####### if parallel #######
    #     tasks = []
    #     for dir_name in tqdm(dir_list):
    #         path_name = os.path.join(args.SORT_PATH, dir_name, target_period)
    #         files = glob.glob(os.path.join(path_name, '*.csv'))
    #         files.sort()
    #         file_name_list = [os.path.basename(f) for f in files if os.path.isfile(f)]
    #         for tmp_NAME in file_name_list:
    #             tasks.append((args, dir_name, path_name, target_period, tmp_NAME))
    #
    #     Parallel(n_jobs=10)(delayed(process_file)(*task) for task in tasks)
    #     #########################
    #
    #
    #     # ####### if single #######
    #     # for dir_name in dir_list:
    #     #     path_name = os.path.join(args.SORT_PATH, dir_name, target_period)
    #     #     files = glob.glob(os.path.join(path_name, '*.csv'))
    #     #     files.sort()
    #     #     file_name_list = [os.path.basename(f) for f in files if os.path.isfile(f)]
    #     #     if len(files) != 0:
    #     #         files.sort()
    #     #         for tmp_NAME in file_name_list:
    #     #             save_path = os.path.join(args.SAMPLE_PATH, dir_name)
    #     #             if not os.path.exists(save_path):
    #     #                 os.mkdir(save_path)
    #     #             save_path = os.path.join(save_path, str(target_period))
    #     #             if not os.path.exists(save_path):
    #     #                 os.mkdir(save_path)
    #     #             save_path = os.path.join(save_path, tmp_NAME)
    #     #             tgt_path = os.path.join(path_name, tmp_NAME)
    #     #             sample_data(args, tgt_path, save_path)
    #     # ########################
    #
    #     eval_stat(args.SAMPLE_PATH, target_period)
    print('done')