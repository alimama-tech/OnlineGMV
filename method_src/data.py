import os
import pickle
from logging import getLogger
import pandas as pd
import ast
import numpy as np
import copy

SECONDS_A_DAY = 60*60*24

logger = getLogger()

def get_pretrain_data_df(args):
    data_path = os.path.join(args.data_path, args.dataset_name)
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, sep="\t")
    logger.info("df head\n%s", df.head())
    logger.info("preprocessing data from %s",args.data_path)
    logger.info("raw DataFrame shape: %s", df.shape)

    if type(df['pay_time'][0]) == str:
        df['pay_time'] = df['pay_time'].apply(ast.literal_eval)

    if type(df['dirpay_amt'][0]) == str:
        df['dirpay_amt'] = df['dirpay_amt'].apply(ast.literal_eval)
    
    df['first_pay_time'] = df['pay_time'].apply(lambda x: x[0] if len(x) > 0 else -1)
    if (df['first_pay_time'] == -1).any():
        logger.info("exist first_pay_time is -1")
    df['first_pay_amt'] = df['dirpay_amt'].apply(lambda x: x[0] if len(x) > 0 else -1)
    if (df['first_pay_amt'] == -1).any():
        logger.info("exist first_pay_amt is -1")
    def safe_sum(lst):
        return sum(pd.to_numeric(i, errors='coerce') or 0 for i in lst)
    df['gmv'] = df['dirpay_amt'].apply(safe_sum)
    if (df['gmv'] == 0).any():
        logger.info("exist gmv is 0")

    click_ts = df['click_time'].to_numpy() # (n, )
    pay_ts = df['first_pay_time'].to_numpy() # (n, )
    first_pay_gmv = df['first_pay_amt'].to_numpy()
    final_gmv = df['gmv'].to_numpy()
    multi_tag = df['multi_tag'].to_numpy() # (n, )
    k = df['total_counts'].to_numpy() # (n, )
  
    columns_to_drop = ['click_time', 'first_pay_time', 'first_pay_amt', 'dirpay_amt', 'pay_time','gmv','multi_tag', 'count', 'total_counts',  'prev_dirpay_amt', 'prev_pay_time']
    df = df.drop(columns=columns_to_drop)
    for c in df.columns:
        df[c] = df[c].fillna(0)
    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, k

def get_pretrain_calibrator_backfill_data_df(args):
    data_path = os.path.join(args.data_path, args.dataset_name)
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, sep="\t")
    logger.info("df head\n%s", df.head())
    logger.info("preprocessing data from %s",args.data_path)
    logger.info("DataFrame shape: %s", df.shape)
    df = df[df['multi_tag'] == 1]
    df = df.drop(columns=['prev_dirpay_amt', 'prev_pay_time'])
    
    df['pay_time'] = df['pay_time'].apply(ast.literal_eval)
    df['dirpay_amt'] = df['dirpay_amt'].apply(ast.literal_eval)
    df['count'] = df['count'].apply(ast.literal_eval)

    df['dirpay_amt'] = df['dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())
    df['gmv'] = df['dirpay_amt'].apply(lambda x: x[-1])
    df = df.explode(['pay_time', 'dirpay_amt', 'count'])
    df = df.reset_index(drop=True)
    
    df['pay_time'] = df['pay_time'].astype(int)
    df['dirpay_amt'] = df['dirpay_amt'].astype(float)
    df['count'] = df['count'].astype(int)

    click_ts = df['click_time'].to_numpy() # (n, )
    pay_ts = df['pay_time'].to_numpy() # (n, )
    first_pay_gmv = df['dirpay_amt'].to_numpy()
    final_gmv = df['gmv'].to_numpy()
    multi_tag = df['multi_tag'].to_numpy() # (n, )
    k = df['total_counts'].to_numpy() # (n, )
    now_count = df['count'].to_numpy() # (n, )
 
    columns_to_drop = ['click_time', 'pay_time', 'dirpay_amt', 'gmv','multi_tag', 'count', 'total_counts']
    df = df.drop(columns=columns_to_drop)

    for c in df.columns:
        df[c] = df[c].fillna(0)

    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, k, now_count

class DataDF(object):
    def __init__(self, features, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, sample_ts=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.first_pay_gmv = copy.deepcopy(first_pay_gmv)
        self.final_gmv = copy.deepcopy(final_gmv)
        self.multi_tag = copy.deepcopy(multi_tag)

        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.first_pay_gmv[mask],
                      self.final_gmv[mask],
                      self.multi_tag[mask],
                      self.sample_ts[mask]
                      )
    
    def sub_test_pay_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.pay_ts >= start_ts,
                              self.pay_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.first_pay_gmv[mask],
                      self.final_gmv[mask],
                      self.multi_tag[mask],
                      self.sample_ts[mask]
                      )
    
    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.first_pay_gmv[idx],
                      self.final_gmv[idx],
                      self.multi_tag[idx],
                      self.sample_ts[idx]
                      )
    
    def form_stream(self):
        observed_df = self.x.copy(deep=True)
        x = observed_df
        sample_ts = self.pay_ts  
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        first_pay_gmv = self.first_pay_gmv 
        final_gmv = self.final_gmv
        multi_tag = self.multi_tag

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda i: sample_ts[i])

        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      first_pay_gmv[idx],
                      final_gmv[idx],
                      multi_tag[idx],
                      sample_ts[idx])
    
    def form_pretrian_bdl_stream(self):
        observed_df = self.x.copy(deep=True)
        x = observed_df 
        sample_ts = self.click_ts + 7 * SECONDS_A_DAY 
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        first_pay_gmv = self.first_pay_gmv 
        final_gmv = self.final_gmv
        multi_tag = self.multi_tag

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda i: sample_ts[i])

        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      first_pay_gmv[idx],
                      final_gmv[idx],
                      multi_tag[idx],
                      sample_ts[idx]
                      )

def get_pretrain_dataset(args):
    dataset_name = args.dataset_name.split('.')[0]
    mode = args.mode
    logger.info("Loading data from cache path: {}".format(args.data_cache_path))
    # cache_path = os.path.join(args.data_cache_path, f"{dataset_name}_{mode}_{args.pkl_remarks}_{args.seed}.pkl")
    cache_path = os.path.join(args.data_cache_path, f"trace_pretrain_all_pay_2025.pkl")
    if os.path.isfile(cache_path):
        logger.info("cache_path {} exists.".format(cache_path))
        logger.info("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        logger.info("cache_path {} does not exist.".format(cache_path))
        logger.info("building datasets...")

        logger.info("use all_pay data")

        df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, _ = get_pretrain_data_df(args)
      
        data = DataDF(df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag)
        train_data = data.sub_days(args.train_start_day,args.train_end_day).shuffle()
        test_data = data.sub_days(args.test_start_day, args.test_end_day)
        logger.info("=======Finish building dataset=======")
        if args.data_cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_data, "test": test_data}, f)
        logger.info("=======Finish saving dataset=======")
        logger.info("train size:{}".format(train_data.x.shape[0]))
        logger.info("test size:{}".format(test_data.x.shape[0]))
    return {
            "train": {
                "x": train_data.x,
                "click_ts": train_data.click_ts,
                "pay_ts": train_data.pay_ts,
                "sample_ts": train_data.sample_ts,
                "first_pay_gmv": train_data.first_pay_gmv,
                "final_gmv": train_data.final_gmv,
                "multi_tag": train_data.multi_tag
            },
            "test": {
                "x": test_data.x,
                "click_ts": test_data.click_ts,
                "pay_ts": test_data.pay_ts,
                "sample_ts": test_data.sample_ts,
                "first_pay_gmv": test_data.first_pay_gmv,
                "final_gmv": test_data.final_gmv,
                "multi_tag": test_data.multi_tag
            }
        }

def get_backfill_data_df(args):  
    data_path = os.path.join(args.data_path, args.dataset_name)
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, sep="\t")
    logger.info("df head\n%s", df.head())
    logger.info("preprocessing data from %s",args.data_path)
    logger.info("raw DataFrame shape: %s", df.shape)
    columns_to_drop = ['prev_dirpay_amt', 'prev_pay_time']
    df = df.drop(columns=columns_to_drop)

    if type(df['pay_time'][0]) == str:
        df['pay_time'] = df['pay_time'].apply(ast.literal_eval)

    if type(df['dirpay_amt'][0]) == str:
        df['dirpay_amt'] = df['dirpay_amt'].apply(ast.literal_eval)

    if type(df['count'][0]) == str:
        df['count'] = df['count'].apply(ast.literal_eval)

    df['dirpay_amt'] = df['dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())
    df['gmv'] = df['dirpay_amt'].apply(lambda x: x[-1])
    df = df.explode(['pay_time', 'dirpay_amt', 'count'])
    df = df.reset_index(drop=True)
    
    df['pay_time'] = df['pay_time'].astype(int)
    df['dirpay_amt'] = df['dirpay_amt'].astype(float)
    df['count'] = df['count'].astype(int)

    click_ts = df['click_time'].to_numpy() # (n, )
    pay_ts = df['pay_time'].to_numpy() # (n, )
    first_pay_gmv = df['dirpay_amt'].to_numpy()
    final_gmv = df['gmv'].to_numpy()
    multi_tag = df['multi_tag'].to_numpy() # (n, )
    k = df['total_counts'].to_numpy() # (n, )
    now_count = df['count'].to_numpy() # (n, )
    columns_to_drop = ['click_time', 'pay_time', 'dirpay_amt', 'gmv','multi_tag', 'count', 'total_counts']
    df = df.drop(columns=columns_to_drop)
    for c in df.columns:
        df[c] = df[c].fillna(0)
    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, k, now_count

def get_stream_dataset(args):
    dataset_name = args.dataset_name
    mode = args.mode
    logger.info("Loading data from cache path: {}".format(args.data_cache_path))
    # cache_path = os.path.join(args.data_cache_path, f"{args.model}_{mode}_{args.pkl_remarks}.pkl")
    if "vanilla" in args.model:
        cache_path = os.path.join(args.data_cache_path, f"vanilla_backfill_{mode}_backfill.pkl")
    else:
        raise ValueError(f"Unsupported dataset setting")
    
    if os.path.isfile(cache_path):
        logger.info("cache_path {} exists.".format(cache_path))
        logger.info("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
    else:
        logger.info("cache_path {} does not exist.".format(cache_path))
        logger.info("building datasets...")
        if "vanilla" in args.model:
            logger.info("train use backfill data")
            train_df, train_click_ts, train_pay_ts, train_first_pay_gmv, train_final_gmv, train_multi_tag, train_k, now_count = get_backfill_data_df(args)
            test_df, test_click_ts, test_pay_ts, test_first_pay_gmv, test_final_gmv, test_multi_tag, test_k = get_pretrain_data_df(args)
            used_train_data = DataDF(train_df, train_click_ts, train_pay_ts, train_first_pay_gmv, train_final_gmv, train_multi_tag,sample_ts=None, k=train_k, now_count=now_count)
            used_test_data = DataDF(test_df, test_click_ts, test_pay_ts, test_first_pay_gmv, test_final_gmv, test_multi_tag,sample_ts=None, k=test_k, now_count=None)
            logger.info("finish loading，start building data...")
            train_data = used_train_data.sub_days(0,82).form_stream()
            test_data = used_test_data.sub_days(50, 82) 
        else:
            raise ValueError(f"Unsupported dataset setting")
        train_stream = []
        test_stream = []
        train_size = 0
        test_size = 0
        for i in np.arange(args.train_start_day, args.train_end_day, args.stream_step):
            train_day = train_data.sub_days(i, i+args.stream_step)
            train_size += train_day.x.shape[0]
            train_stream.append({
                "x":train_day.x,
                "click_ts": train_day.click_ts,
                "pay_ts": train_day.pay_ts,
                "sample_ts": train_day.sample_ts,
                "first_pay_gmv": train_day.first_pay_gmv,
                "final_gmv": train_day.final_gmv,
                "multi_tag": train_day.multi_tag,
                "k": train_day.k,
                "now_count": train_day.now_count,
            })
        for i in np.arange(args.test_start_day, args.test_end_day, args.stream_step):
            test_day = test_data.sub_days(i, i+args.stream_step)
            test_size += test_day.x.shape[0]
            test_stream.append({
                "x":test_day.x,
                "click_ts": test_day.click_ts,
                "pay_ts": test_day.pay_ts,
                "sample_ts": test_day.sample_ts,
                "first_pay_gmv": test_day.first_pay_gmv,
                "final_gmv": test_day.final_gmv,
                "multi_tag": test_day.multi_tag,
                "k": test_day.k,
                "now_count": test_day.now_count,
            })
  
        logger.info("finishi building, start saving...")
        if args.data_cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_stream, "test": test_stream}, f)
        logger.info("=======Finish saving dataset=======")
        logger.info(f"train size:{train_size}")
        logger.info(f"test size:{test_size}")

    return train_stream, test_stream

def get_pretrain_bdl_stream_dataset(args):
    mode = args.mode
    cache_path = os.path.join(args.data_cache_path, f"pretrain_bdl_{mode}_all_pay.pkl")
    logger.info("Loading data from cache path: {}".format(cache_path))
    if os.path.isfile(cache_path):
        logger.info("cache_path {} exists.".format(cache_path))
        logger.info("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
    else:
        logger.info("cache_path {} does not exist.".format(cache_path))
        logger.info("building datasets...")
        df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, _ = get_pretrain_data_df(args)
        logger.info("finish loading，strat build pretrain_bdl data...")
        data = DataDF(df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag)
        train_data = data.sub_days(0, 82).form_pretrian_bdl_stream() 
        test_data = data.sub_days(50, 82)
        train_stream = []
        test_stream = []
        train_size = 0
        test_size = 0

        for i in np.arange(args.train_start_day, args.train_end_day, args.bdl_stream_step):
            train_day = train_data.sub_days(i, i+args.bdl_stream_step)
            train_size += train_day.x.shape[0]
            train_stream.append({
                "x":train_day.x,
                "click_ts": train_day.click_ts,
                "pay_ts": train_day.pay_ts,
                "sample_ts": train_day.sample_ts,
                "first_pay_gmv": train_day.first_pay_gmv,
                "final_gmv": train_day.final_gmv,
                "multi_tag": train_day.multi_tag 
            })
        for i in np.arange(args.test_start_day, args.test_end_day, args.stream_step):
            test_day = test_data.sub_days(i, i+args.stream_step)
            test_size += test_day.x.shape[0]
            test_stream.append({
                "x":test_day.x,
                "click_ts": test_day.click_ts,
                "pay_ts": test_day.pay_ts,
                "sample_ts": test_day.sample_ts,
                "first_pay_gmv": test_day.first_pay_gmv,
                "final_gmv": test_day.final_gmv,
                "multi_tag": test_day.multi_tag
            })

        logger.info("finish building, start saving...")
        if args.data_cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_stream, "test": test_stream}, f)
        logger.info("=======Finish saving dataset=======")
        logger.info(f"train size:{train_size}, train days:{len(train_stream)}")
        logger.info(f"test size:{test_size}, test days:{len(test_stream)}")

    return train_stream, test_stream

class CalibratorDataDF(object):
    def __init__(self, features, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, sample_ts=None, k=None, now_count=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.first_pay_gmv = copy.deepcopy(first_pay_gmv)
        self.final_gmv = copy.deepcopy(final_gmv)
        self.multi_tag = copy.deepcopy(multi_tag)

        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        
        if k is not None:
            self.k = copy.deepcopy(k)
        else:
            raise ValueError("k is None")
        
        if now_count is not None:
            self.now_count = copy.deepcopy(now_count)
        else:
            raise ValueError("now_count is None")

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return CalibratorDataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.first_pay_gmv[mask],
                      self.final_gmv[mask],
                      self.multi_tag[mask],
                      self.sample_ts[mask],
                      self.k[mask],
                      self.now_count[mask]
                      )
    
    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return CalibratorDataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.first_pay_gmv[idx],
                      self.final_gmv[idx],
                      self.multi_tag[idx],
                      self.sample_ts[idx],
                      self.k[idx],
                      self.now_count[idx]
                      )

def get_pretrain_calibrator_dataset(args):
    dataset_name = args.dataset_name.split('.')[0]
    mode = args.mode
    logger.info("Loading data from cache path: {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"calibrator_with_time_pretrain.pkl")
    if os.path.isfile(cache_path):
        logger.info("cache_path {} exists.".format(cache_path))
        logger.info("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        logger.info("cache_path {} does not exist.".format(cache_path))
        logger.info("building datasets...")

        logger.info("use all_pay data")

        df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, k, now_count = get_pretrain_calibrator_backfill_data_df(args)
      
        data = CalibratorDataDF(df, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag,sample_ts=None, k=k, now_count=now_count)
        train_data = data.sub_days(args.train_start_day,args.train_end_day).shuffle()
        test_data = data.sub_days(args.test_start_day, args.test_end_day)
        logger.info("=======Finish building dataset=======")
        if args.data_cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_data, "test": test_data}, f)
        logger.info("=======Finish saving dataset=======")
        logger.info("train size:{}".format(train_data.x.shape[0]))
        logger.info("test size:{}".format(test_data.x.shape[0]))
    return {
            "train": {
                "x": train_data.x,
                "click_ts": train_data.click_ts,
                "pay_ts": train_data.pay_ts,
                "sample_ts": train_data.sample_ts,
                "first_pay_gmv": train_data.first_pay_gmv,
                "final_gmv": train_data.final_gmv,
                "multi_tag": train_data.multi_tag,
                "k": train_data.k,
                "now_count": train_data.now_count
            },
            "test": {
                "x": test_data.x,
                "click_ts": test_data.click_ts,
                "pay_ts": test_data.pay_ts,
                "sample_ts": test_data.sample_ts,
                "first_pay_gmv": test_data.first_pay_gmv,
                "final_gmv": test_data.final_gmv,
                "multi_tag": test_data.multi_tag,
                "k": test_data.k,
                "now_count": test_data.now_count
            }
        }

############reader dataset##################

def get_backfill_with_prev_data_df(args): 
    logger.info("Loading data from %s", args.data_path)
    data_path = os.path.join(args.data_path, args.dataset_name)
    df = pd.read_csv(data_path, sep="\t")
    logger.info("df head\n%s", df.head())
    logger.info("preprocessing data from %s",args.data_path)
    logger.info("raw DataFrame shape: %s", df.shape)

    if type(df['pay_time'][0]) == str:
        df['pay_time'] = df['pay_time'].apply(ast.literal_eval)

    if type(df['dirpay_amt'][0]) == str:
        df['dirpay_amt'] = df['dirpay_amt'].apply(ast.literal_eval)
    
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].apply(ast.literal_eval)
    df['prev_pay_time'] = df['prev_pay_time'].apply(ast.literal_eval)

    if type(df['count'][0]) == str:
        df['count'] = df['count'].apply(ast.literal_eval)

    df['dirpay_amt'] = df['dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())
    df['gmv'] = df['dirpay_amt'].apply(lambda x: x[-1])

    df_final_gmv = df.copy(deep=True)
    df_final_gmv['prev_pay_time'] = df_final_gmv['pay_time'].apply(lambda x: x[-1])
    df_final_gmv['prev_dirpay_amt'] = df_final_gmv['dirpay_amt'].apply(lambda x: x[-1])
    df_final_gmv['pay_time'] = df_final_gmv['click_time'] + 7 * SECONDS_A_DAY
    df_final_gmv['dirpay_amt'] = df_final_gmv['dirpay_amt'].apply(lambda x: x[-1])
    df_final_gmv['count'] = df_final_gmv['count'].apply(lambda x: x[-1]) + 1
    

    df = df.explode(['pay_time', 'dirpay_amt', 'count', 'prev_dirpay_amt', 'prev_pay_time'])
    df = df.reset_index(drop=True)
    df['pay_time'] = df['pay_time'].astype(int)
    df['dirpay_amt'] = df['dirpay_amt'].astype(float)
    df['count'] = df['count'].astype(int)
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].astype(float)
    df['prev_pay_time'] = df['prev_pay_time'].astype(int)
    logger.info(f'data process done, now data df:\n {df.head()}')

    df_final_gmv['pay_time'] = df_final_gmv['pay_time'].astype(int)
    df_final_gmv['dirpay_amt'] = df_final_gmv['dirpay_amt'].astype(float)
    df_final_gmv['count'] = df_final_gmv['count'].astype(int)
    df_final_gmv['prev_dirpay_amt'] = df_final_gmv['prev_dirpay_amt'].astype(float)
    df_final_gmv['prev_pay_time'] = df_final_gmv['prev_pay_time'].astype(int)
    logger.info(f'data_final_gmv process done, now data df_final_gmv:\n {df_final_gmv.head()}')
    df = pd.concat([df, df_final_gmv], axis=0, ignore_index=True, sort=False)

    click_ts = df['click_time'].to_numpy() # (n, )
    now_pay_ts = df['pay_time'].to_numpy() # (n, )
    now_pay_gmv = df['dirpay_amt'].to_numpy()
    final_gmv = df['gmv'].to_numpy()
    multi_tag = df['multi_tag'].to_numpy() # (n, )
    k = df['total_counts'].to_numpy() # (n, )
    now_count = df['count'].to_numpy() # (n, )
    prev_pay_gmv = df['prev_dirpay_amt'].to_numpy() # (n, )
    prev_pay_ts = df['prev_pay_time'].to_numpy() # (n, )

    columns_to_drop = ['click_time', 'pay_time', 'dirpay_amt', 'gmv','multi_tag', 'count', 'total_counts', 'prev_dirpay_amt', 'prev_pay_time']
    df = df.drop(columns=columns_to_drop)
    for c in df.columns:
        df[c] = df[c].fillna(0)
    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, now_pay_ts, now_pay_gmv, final_gmv, multi_tag, k, now_count, prev_pay_gmv, prev_pay_ts

def get_backfill_with_prev_test_data_df(args): 
    logger.info("Loading data from %s", args.data_path)
    data_path = os.path.join(args.data_path, args.dataset_name)
    df = pd.read_csv(data_path, sep="\t")
    logger.info("df head\n%s", df.head())
    logger.info("preprocessing data from %s",args.data_path)
    logger.info("raw DataFrame shape: %s", df.shape)

    if type(df['pay_time'][0]) == str:
        df['pay_time'] = df['pay_time'].apply(ast.literal_eval)

    if type(df['dirpay_amt'][0]) == str:
        df['dirpay_amt'] = df['dirpay_amt'].apply(ast.literal_eval)
    
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].apply(ast.literal_eval)
    df['prev_pay_time'] = df['prev_pay_time'].apply(ast.literal_eval)

    if type(df['count'][0]) == str:
        df['count'] = df['count'].apply(ast.literal_eval)

    df['dirpay_amt'] = df['dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].apply(lambda x: np.cumsum(x).tolist())

    df['gmv'] = df['dirpay_amt'].apply(lambda x: x[-1])
    
    df['pay_time'] = df['pay_time'].apply(lambda x: x[0]).astype(int)
    df['dirpay_amt'] = df['dirpay_amt'].apply(lambda x: x[0]).astype(float)
    df['count'] = df['count'].apply(lambda x: x[0]).astype(int)
    df['prev_dirpay_amt'] = df['prev_dirpay_amt'].apply(lambda x: x[0]).astype(float)
    df['prev_pay_time'] = df['prev_pay_time'].apply(lambda x: x[0]).astype(int)
    logger.info(f'data process done, now data df:\n {df.head()}')

    click_ts = df['click_time'].to_numpy() # (n, )
    now_pay_ts = df['pay_time'].to_numpy() # (n, )
    now_pay_gmv = df['dirpay_amt'].to_numpy()
    final_gmv = df['gmv'].to_numpy()
    multi_tag = df['multi_tag'].to_numpy() # (n, )
    k = df['total_counts'].to_numpy() # (n, )
    now_count = df['count'].to_numpy() # (n, )
    prev_pay_gmv = df['prev_dirpay_amt'].to_numpy() # (n, )
    prev_pay_ts = df['prev_pay_time'].to_numpy() # (n, )

    columns_to_drop = ['click_time', 'pay_time', 'dirpay_amt', 'gmv','multi_tag', 'count', 'total_counts', 'prev_dirpay_amt', 'prev_pay_time']
    df = df.drop(columns=columns_to_drop)
    for c in df.columns:
        df[c] = df[c].fillna(0)
    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, now_pay_ts, now_pay_gmv, final_gmv, multi_tag, k, now_count, prev_pay_gmv, prev_pay_ts

def get_stream_with_prev_dataset(args):
    dataset_name = args.dataset_name
    mode = args.mode
    logger.info("Loading data from cache path: {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"vanilla_backfill_{mode}_backfill_with_prev_attri_back.pkl")
    if os.path.isfile(cache_path):
        logger.info("cache_path {} exists.".format(cache_path))
        logger.info("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
    else:
        logger.info("cache_path {} does not exist.".format(cache_path))
        logger.info("building datasets...")
        df, click_ts, now_pay_ts, now_pay_gmv, final_gmv, multi_tag, \
            k, now_count, prev_pay_gmv, prev_pay_ts = get_backfill_with_prev_data_df(args)
        
        test_df, test_click_ts, test_now_pay_ts, test_now_pay_gmv, test_final_gmv, test_multi_tag, \
            test_k, test_now_count, test_prev_pay_gmv, test_prev_pay_ts = get_backfill_with_prev_test_data_df(args)
        logger.info("finish loading，start building data...")
        data = DataDFwithPrev(df, click_ts, now_pay_ts, now_pay_gmv, final_gmv, multi_tag, sample_ts=None, k=k, now_count=now_count, prev_pay_gmv=prev_pay_gmv, prev_pay_ts=prev_pay_ts)
        test_data = DataDFwithPrev(test_df, test_click_ts, test_now_pay_ts, test_now_pay_gmv, test_final_gmv, test_multi_tag, sample_ts=None, k=test_k, now_count=test_now_count, prev_pay_gmv=test_prev_pay_gmv, prev_pay_ts=test_prev_pay_ts)
        train_data = data.sub_days(0, 82).form_stream()
        test_data = test_data.sub_days(50, 82)
        train_stream = []
        test_stream = []
        train_size = 0
        test_size = 0
        for i in np.arange(args.train_start_day, args.train_end_day, args.stream_step):
            train_day = train_data.sub_days(i, i+args.stream_step)
            train_size += train_day.x.shape[0]
            train_stream.append({
                "x":train_day.x,
                "click_ts": train_day.click_ts,
                "pay_ts": train_day.pay_ts,
                "sample_ts": train_day.sample_ts,
                "first_pay_gmv": train_day.first_pay_gmv,
                "final_gmv": train_day.final_gmv,
                "multi_tag": train_day.multi_tag,
                "k": train_day.k,
                "now_count": train_day.now_count,
                "prev_pay_gmv": train_day.prev_pay_gmv,
                "prev_pay_ts": train_day.prev_pay_ts
            })
        for i in np.arange(args.test_start_day, args.test_end_day, args.stream_step):
            test_day = test_data.sub_days(i, i+args.stream_step)
            test_size += test_day.x.shape[0]
            test_stream.append({
                "x":test_day.x,
                "click_ts": test_day.click_ts,
                "pay_ts": test_day.pay_ts,
                "sample_ts": test_day.sample_ts,
                "first_pay_gmv": test_day.first_pay_gmv,
                "final_gmv": test_day.final_gmv,
                "multi_tag": test_day.multi_tag,
                "k": test_day.k,
                "now_count": test_day.now_count,
                "prev_pay_gmv": test_day.prev_pay_gmv,
                "prev_pay_ts": test_day.prev_pay_ts
            })
            
        logger.info("finish building，start saving...")
        if args.data_cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_stream, "test": test_stream}, f)
        logger.info("=======Finish saving dataset=======")
        logger.info(f"train size:{train_size}")
        logger.info(f"test size:{test_size}")
    return train_stream, test_stream


class DataDFwithPrev(object):
    def __init__(self, features, click_ts, pay_ts, first_pay_gmv, final_gmv, multi_tag, sample_ts=None, k=None, now_count=None, prev_pay_gmv=None,prev_pay_ts=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.first_pay_gmv = copy.deepcopy(first_pay_gmv)
        self.final_gmv = copy.deepcopy(final_gmv)
        self.multi_tag = copy.deepcopy(multi_tag)

        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        
        if k is not None:
            self.k = copy.deepcopy(k)
        else:
            raise ValueError("k is None")

        if now_count is not None:
            self.now_count = copy.deepcopy(now_count)
        else:
            raise ValueError("now_count is None")

        if prev_pay_gmv is not None:
            self.prev_pay_gmv = copy.deepcopy(prev_pay_gmv)
        else:
            raise ValueError("prev_pay_gmv is None")

        if prev_pay_ts is not None:
            self.prev_pay_ts = copy.deepcopy(prev_pay_ts)
        else:
            raise ValueError("prev_pay_ts is None")

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDFwithPrev(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.first_pay_gmv[mask],
                      self.final_gmv[mask],
                      self.multi_tag[mask],
                      self.sample_ts[mask],
                      self.k[mask],
                      self.now_count[mask],
                      self.prev_pay_gmv[mask],
                      self.prev_pay_ts[mask]
                      )
    def form_stream(self):
        observed_df = self.x.copy(deep=True)
        x = observed_df
        sample_ts = self.pay_ts  
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        first_pay_gmv = self.first_pay_gmv 
        final_gmv = self.final_gmv
        multi_tag = self.multi_tag
        k = self.k
        now_count = self.now_count
        prev_pay_gmv = self.prev_pay_gmv
        prev_pay_ts = self.prev_pay_ts

        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda i: sample_ts[i])

        return DataDFwithPrev(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      first_pay_gmv[idx],
                      final_gmv[idx],
                      multi_tag[idx],
                      sample_ts[idx],
                      k[idx],
                      now_count[idx],
                      prev_pay_gmv[idx],
                      prev_pay_ts[idx])
