from joblib import Parallel, delayed
from io import StringIO
import os
import glob
import pandas as pd
from tqdm import tqdm
from config import *
# from utils import *


def sort_rawdata(args, raw_path, period):
    def process_file(file_path, args, period):
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        csv_string = binary_data.decode('utf-8').replace('\x00', '')

        na_values = ['\\N', r'\N']
        df = pd.read_csv(StringIO(csv_string), header=None, na_values=na_values, keep_default_na=True)

        if df.shape[1] == 21:
            df.columns = set_header('202111')
        elif df.shape[1] == 22:
            df.columns = set_header('202202')
        else:
            df.columns = set_header('202402')
        # df.columns = args.HEADER_LIST
        df['cSenID'] = df['cSenID'].astype(str).str[-8:].astype(int)
        unique_num_id = df['cSenID'].unique()

        for tmp_val in unique_num_id:
            tmp_id_df = df.loc[df['cSenID'] == tmp_val]
            save_path = os.path.join(args.SORT_PATH, str(tmp_val))
            os.makedirs(save_path, exist_ok=True)

            save_path = os.path.join(save_path, str(period))
            os.makedirs(save_path, exist_ok=True)

            save_file_prefix = str(tmp_val)
            save_file = f"{save_file_prefix}_{tmp_id_df['cSenDate'].unique()[0]}_.csv"
            save_file_path = os.path.join(save_path, save_file)

            # if os.path.exists(save_file_path):
            #     print(f'[Already Sorted] {save_file}')
            # else:
            tmp_id_df.to_csv(save_file_path, index=False)

    print('Sort data by ID')

    files_raw = glob.glob(os.path.join(raw_path, '*.csv'))
    files_raw.sort()

    ### if parallel
    Parallel(n_jobs=-1)(
        delayed(process_file)(file_path, args, period) for file_path in tqdm(files_raw)
    )


    '''
    ### if single
    file_name_list = [os.path.basename(f) for f in files_raw if os.path.isfile(f)]
    for FILE_NAME in tqdm(file_name_list):
        file_path = os.path.join(raw_path, FILE_NAME)

        with open(file_path, 'rb') as f:
            binary_data = f.read()
        csv_string = binary_data.decode('utf-8').replace('\x00', '')

        df = pd.read_csv(StringIO(csv_string), header=None, na_values=na_values, keep_default_na=True)
        if df.shape[1] == 21:
            df.columns = set_header('202201')
        elif df.shape[1] == 22:
            df.columns = set_header('202202')
        # df.columns = args.HEADER_LIST
        df['cSenID'] = df['cSenID'].apply(lambda x: int(str(x)[-8:]))
        unique_num_id = df['cSenID'].unique()
        for tmp_val in unique_num_id:
            tmp_id_df = df.loc[df['cSenID'] == tmp_val]
            save_path = args.SORT_PATH + str(tmp_val) + '/'
            if not os.path.exists(args.SORT_PATH + str(tmp_val)):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, str(period))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_file_prefix = str(tmp_val)
            save_file = save_file_prefix + '_' + tmp_id_df['cSenDate'].unique()[0] + "_.csv"
            if os.path.exists(os.path.join(save_path, save_file)):
                print('[Already Sorted] {}'.format(save_file))
            else:
                tmp_id_df.to_csv(os.path.join(save_path, save_file), index=False)
    '''


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_file(args, dir_name, path_name, target_period, tmp_NAME):
    save_path = os.path.join(args.SAMPLE_PATH, dir_name, str(target_period))
    create_dir_if_not_exists(save_path)
    save_path = os.path.join(save_path, tmp_NAME)
    tgt_path = os.path.join(path_name, tmp_NAME)
    sample_data(args, tgt_path, save_path)


if __name__ == "__main__":
    args = set_parameters()

    # target_period = args.COLLECT_PERIOD[2]
    target_index = ['202406', '202407']
    period_list = [i for i in target_index]
    print('Data Preprocessing for {}'.format(period_list))

    sort_rawdata(args)

    for target_period in period_list:
        print('Period: {}'.format(target_period))

        # set folder of rawdata files name as 'raw_202202'
        DATA_PATH = 'data/' + 'raw_' + target_period + '/'

        ####### Filtered by USER ID #######
        sort_rawdata(args, DATA_PATH, target_period)
        # eval_stat(args.SORT_PATH, target_period)

        #### sample data with velocity
        dir_list = os.listdir(args.SORT_PATH)
        dir_list.sort()

        ####### if parallel #######
        tasks = []
        for dir_name in tqdm(dir_list):
            path_name = os.path.join(args.SORT_PATH, dir_name, target_period)
            files = glob.glob(os.path.join(path_name, '*.csv'))
            files.sort()
            file_name_list = [os.path.basename(f) for f in files if os.path.isfile(f)]
            for tmp_NAME in file_name_list:
                tasks.append((args, dir_name, path_name, target_period, tmp_NAME))

        Parallel(n_jobs=-1)(delayed(process_file)(*task) for task in tasks)
        #########################


        # ####### if single #######
        # for dir_name in dir_list:
        #     path_name = os.path.join(args.SORT_PATH, dir_name, target_period)
        #     files = glob.glob(os.path.join(path_name, '*.csv'))
        #     files.sort()
        #     file_name_list = [os.path.basename(f) for f in files if os.path.isfile(f)]
        #     if len(files) != 0:
        #         files.sort()
        #         for tmp_NAME in file_name_list:
        #             save_path = os.path.join(args.SAMPLE_PATH, dir_name)
        #             if not os.path.exists(save_path):
        #                 os.mkdir(save_path)
        #             save_path = os.path.join(save_path, str(target_period))
        #             if not os.path.exists(save_path):
        #                 os.mkdir(save_path)
        #             save_path = os.path.join(save_path, tmp_NAME)
        #             tgt_path = os.path.join(path_name, tmp_NAME)
        #             sample_data(args, tgt_path, save_path)
        # ########################

        # eval_stat(args.SAMPLE_PATH, target_period)
    print('done')