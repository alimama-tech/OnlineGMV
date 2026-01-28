

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from logging import getLogger

logger = getLogger()

class TrainingDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)

        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"TrainingDataset size: {size}")
        logger.info(f"TrainingDataset multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx]
                }
    
class EvaluationDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)

        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"EvaluationDataset size: {size}")
        logger.info(f"EvaluationDataset multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx]
                }
    
class StreamDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)
        self.k = torch.tensor(datadict['k'], dtype=torch.long)
        self.now_count = torch.tensor(datadict['now_count'], dtype=torch.long) 

        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"data_size: {size}")
        logger.info(f"multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx],
                "k": self.k[idx],
                "now_count": self.now_count[idx]
                }
    
class StreamDataLoader:
    def __init__(self, stream_dataset, batch_size=1024, shuffle=False, num_workers=4):
        self.stream_dataset = stream_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    def __len__(self):
        return len(self.stream_dataset)
    def get_day_dataloader(self, day):
        day_dataset = StreamDataset(self.stream_dataset[day])
        return DataLoader(day_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class Pretrain_BDL_StreamDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)
    
        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"data_size: {size}")
        logger.info(f"multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx],
                }

class Pretrain_BDL_StreamDataLoader:
    def __init__(self, stream_dataset, batch_size=1024, shuffle=False, num_workers=4):
        self.stream_dataset = stream_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    def __len__(self):
        return len(self.stream_dataset)
    def get_day_dataloader(self, day):
        day_dataset = Pretrain_BDL_StreamDataset(self.stream_dataset[day])
        return DataLoader(day_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
class CalibratorTrainingDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)
        self.k = torch.tensor(datadict['k'], dtype=torch.long)
        self.now_count = torch.tensor(datadict['now_count'], dtype=torch.long)

        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"TrainingDataset size: {size}")
        logger.info(f"TrainingDataset multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx],
                "k": self.k[idx],
                "now_count": self.now_count[idx]
                }
    
class CalibratorEvaluationDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)
        self.k = torch.tensor(datadict['k'], dtype=torch.long)
        self.now_count = torch.tensor(datadict['now_count'], dtype=torch.long)

        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"EvaluationDataset size: {size}")
        logger.info(f"EvaluationDataset multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx],
                "k": self.k[idx],
                "now_count": self.now_count[idx]
                }   
    
class With_GA_StreamDataset(Dataset):
    def __init__(self, datadict):
        self.features = torch.tensor(datadict['x'].to_numpy(), dtype=torch.long)
        self.click_ts = torch.tensor(datadict['click_ts'], dtype=torch.float32)
        self.pay_ts = torch.tensor(datadict['pay_ts'], dtype=torch.float32)
        self.first_pay_gmv = torch.tensor(datadict['first_pay_gmv'], dtype=torch.float32)
        self.final_gmv = torch.tensor(datadict['final_gmv'], dtype=torch.float32)
        self.multi_tag = torch.tensor(datadict['multi_tag'], dtype=torch.long)
        self.k = torch.tensor(datadict['k'], dtype=torch.long)
        self.now_count = torch.tensor(datadict['now_count'], dtype=torch.long)
        self.prev_pay_gmv = torch.tensor(datadict['prev_pay_gmv'], dtype=torch.float32)
        self.prev_pay_ts = torch.tensor(datadict['prev_pay_ts'], dtype=torch.float32)
    
        size = len(self.features)
        multi_pay_size = sum(self.multi_tag)
        logger.info(f"data_size: {size}")
        logger.info(f"multi_pay_size: {multi_pay_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx],
                "click_ts": self.click_ts[idx],
                "pay_ts": self.pay_ts[idx],
                "first_pay_gmv": self.first_pay_gmv[idx],
                "final_gmv": self.final_gmv[idx],
                "multi_tag": self.multi_tag[idx],
                "k": self.k[idx],
                "now_count": self.now_count[idx],
                "prev_pay_gmv": self.prev_pay_gmv[idx],
                "prev_pay_ts": self.prev_pay_ts[idx]
                }

class With_GA_StreamDataLoader:
    def __init__(self, stream_dataset, batch_size=1024, shuffle=False, num_workers=4):
        self.stream_dataset = stream_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    def __len__(self):
        return len(self.stream_dataset)
    def get_day_dataloader(self, day):
        day_dataset = With_GA_StreamDataset(self.stream_dataset[day])
        return DataLoader(day_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
