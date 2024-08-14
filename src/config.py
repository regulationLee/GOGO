import os
import argparse

def set_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sm', "--sample_minute", type=int, default=360)  # SAMPLE_MINUTE
    parser.add_argument('-dm', "--drop_minute", type=int, default=10)  # DROP_MINUTE
    parser.add_argument('-st', "--stop_threshold", type=float, default=4.0)  # STOP_THRESHOLD
    parser.add_argument('-ss', "--stop_sec", type=int, default=30)  # STOP_SEC
    parser.add_argument('-im', "--interval_min", type=int, default=60)  # interval_min
    parser.add_argument('-ud', "--update_decay", type=int, default=5)  # update_decay

    parser.add_argument('-eda', "--set_threshold", action='store_true')

    args = parser.parse_args()
    args.sample_list = [
        '24341577',
        '26674039',
        '29550344',
        '41290104',
        '41893039',
        '44402381',
        '55716450',
        '75218248',
        '75925582',
        '88344483',
        '92530466',
        '93465680'
    ]
    args.accidents = [
        '41290104',
        '26674039',
        '41893039',
        '75218248'
    ] # '57510162',

    args.non_accidents = [
        '24341577',
        '29550344',
        '44402381',
        '55495892',
        '55507348',
        '55716450',
        '75925582',
        '88344483',
        '91609436',
        '92530466',
        '93465680'
    ]  # '57510162',

    # except records that has alignment issue
    args.exception = {
        'cSenAccX': [55507348, 91609436],
        'cSenAccY': [55507348, 91609436],
        'cSenAccZ': [],
        'cSenGyrX': [],
        'cSenGyrY': [],
        'cSenGyrZ': [],
        'cSenAngX': [55507348, 91609436],
        'cSenAngY': [],
        'cSenAngZ': [],
        'AccNorm': [],
        'zigzag': []
    }

    args.COLLECT_PERIOD = ['202111', '202201', '202202', '202203', '202204', '202205', '202402']

    # ignore the data that has the number of dataframe for a day less than drop_period
    args.drop_period = 10 * 60 * args.drop_minute  # drop_period
    # sample_period: frame interval for computing raw safety index
    args.sample_period = 10 * 60 * args.sample_minute
    # args.sample_period = 'per_day'
    args.stop_period = 10 * args.stop_sec

    args.tgt_abnormal_ratio = [0.25, 0.15]  # [normal, abnormal]

    args.interval = int(args.interval_min / args.sample_minute)
    args.index_threshold = [0.1, 0.8] # abnormal ratio: counting abnormal frame in 36,000 frame

    # args.RAW_PATH = '../data/RAWDATA/'
    # args.SORT_PATH = '../data/SORTED/'
    # args.SAMPLE_PATH = '../data/SAMPLED/'
    # args.ACCIDENT_PATH = '../data/raw_accident/'
    # args.EDA_PATH = '../data/EDA/'
    # args.TRAIN_PATH = '../data/DATASET/'

    # Ubuntu
    args.RAW_PATH = '/home/glee/sakak/data/GOGO/' + 'RAWDATA/'
    args.SORT_PATH = '/home/glee/sakak/data/GOGO/' + 'SORTED/'
    args.SAMPLE_PATH = '/home/glee/sakak/data/GOGO/' + 'SAMPLED/'
    args.ACCIDENT_PATH = '/home/glee/sakak/data/GOGO/' + 'raw_accident/'
    args.EDA_PATH = '/home/glee/sakak/data/GOGO/' + 'EDA/'
    args.TRAIN_PATH = '/home/glee/sakak/data/GOGO/' + 'DATASET/'

    if not os.path.exists(args.SORT_PATH):
        os.mkdir(args.SORT_PATH)
    if not os.path.exists(args.SAMPLE_PATH):
        os.mkdir(args.SAMPLE_PATH)
    if not os.path.exists(args.EDA_PATH):
        os.mkdir(args.EDA_PATH)
    if not os.path.exists(args.TRAIN_PATH):
        os.mkdir(args.TRAIN_PATH)

    return args


def set_header(target_period):
    BASIC_HEADER = ['cSenID', 'cSenDate', 'cSenTime', 'cSenType',
                    'cSenAccX', 'cSenAccY', 'cSenAccZ',
                    'cSenGyrX', 'cSenGyrY', 'cSenGyrZ',
                    'cSenAngX', 'cSenAngY', 'cSenAngZ']
    GPS_HEADER = ['gpsLatitude', 'gpsLongitude', 'gpsAltitude', 'gpsSpeed']
    ADD_SEN_HEADER = ['cSenAccX2', 'cSenAccY2', 'cSenAccZ2', 'cSenMageX', 'cSenMageY', 'cSenMageZ']
    TAIL_HEADER = ['datelog', 'riskChkLevel_1', 'riskChkLevel_2']

    if target_period == '202111':
        HEADER_LIST = BASIC_HEADER + ['cSenTemp'] + GPS_HEADER + TAIL_HEADER
    elif target_period == '202202':
        HEADER_LIST = BASIC_HEADER + ['cSenTemp'] + GPS_HEADER + ['movingDistance'] + TAIL_HEADER
    elif target_period == '202212':
        HEADER_LIST = BASIC_HEADER + ['cSenTemp'] + GPS_HEADER + ['movingDistance'] + TAIL_HEADER + ['cDriverId']
    elif target_period == '202402':
        HEADER_LIST = (BASIC_HEADER + ADD_SEN_HEADER + ['cSenTemp']
                       + GPS_HEADER + ['movingDistance'] + TAIL_HEADER + ['cDriverId'])

    return HEADER_LIST


def load_threshold(args, tgt_feat):
    if tgt_feat == 'cSenAccX':
        if args.tgt_abnormal_ratio == [0.2, 0.2]:
            preset_threshold = [0.1661, -0.0366]
            abnormal_ratio = [20.436, 36.146]
        elif args.tgt_abnormal_ratio == [0.3, 0.3]:
            preset_threshold = [0.1154, -0.0366]
            abnormal_ratio = [30.974, 36.146]
        elif args.tgt_abnormal_ratio == [0.4, 0.4]:
            preset_threshold = [0.0901, -0.0366]
            abnormal_ratio = [39.137, 36.146]
    else:
        preset_threshold = 0
        abnormal_ratio = 0

    return preset_threshold, abnormal_ratio
