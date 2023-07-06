import os
import torch
import numpy as np

# 1. load traffic flow dataset
def load_st_dataset(dataset, input_dim=1):
    if dataset == 'PEMSD3':
        data_path = os.path.join('../data/PEMSD3/PEMSD3.npz')
        # only the first dimension, traffic flow data
        data = np.load(data_path)['data'][:, :, :input_dim]
    elif dataset == 'PEMSD4':
        data_path = os.path.join('../data/PEMSD4/PEMSD4.npz')
        data = np.load(data_path)['data'][:, :, :input_dim]
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PEMSD7/PEMSD7.npz')
        data = np.load(data_path)['data'][:, :, :input_dim]
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PEMSD8/PEMSD8.npz')
        data = np.load(data_path)['data'][:, :, :input_dim]
    else:
        raise ValueError
    print("Load %s Dataset shaped: " % dataset, data.shape)
    return data  # output shape: (B, N, D)


# 2. data normalization
# ***********************************************************************************
class NScaler(object):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return data * self.std + self.mean


class MinMax01Scaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return  ((data + 1.) / 2.) * (self.max - self.min) + self.min


def normalize_dataset(data, normalizer):
    if normalizer == 'max01':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax01 Normalization", minimum, maximum)
    elif normalizer == 'max11':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax11Scaler()
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax11 Normalization", minimum, maximum)
    elif normalizer == 'std':
        mean = data.mean()
        std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print("Normalize the dataset by StandardScaler Normalization", mean, std)
    elif normalizer == None:
        scaler = NScaler()
        data = scaler.transform(data)
        print("Does not normalize the dataset")
    else:
        raise ValueError
    return data, scaler
# ***********************************************************************************


# 3. dataset split(training set, verification set, test set)
def split_data_by_days(data, val_days, test_days, interval=60):
    """
    data: (B, N, D)
    val_days:
    test_days:
    interval: interval (5, 15, 30) minutes
    """
    T = int((24 * 60) / interval)
    test_data = data[-T * test_days:]
    val_data = data[-T * (val_days + test_days) : -T * test_days]
    train_data = data[:-T * (val_days + test_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    len_data = data.shape[0]
    test_data = data[-int(len_data * test_ratio):]
    val_data =  data[-int(len_data * (val_ratio + test_ratio)) : -int(len_data * test_ratio)]
    train_data = data[:-int(len_data * (val_ratio + test_ratio))]
    return train_data, val_data, test_data


# 4. sliding window sampling
def Add_Window_Horizon(data, window=12, horizon=12, single=False):
    """
    :param data shape: (B, N, D)
    :param window: 
    :param horizon:
    :return: X is (B, W, N, D), Y is (B, H, N, D)
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []       # windows
    Y = []       # horizon
    index = 0    # 起始索引
    if single:
        # 单步预测, horizon = 1
        while index < end_index:
            X.append(data[index : index + window])
            Y.append(data[index + window + horizon - 1 : index + window + horizon])
            index = index + 1
    else:
        # 多步预测, horizon > 1
        while index < end_index:
            X.append(data[index : index + window])
            Y.append(data[index + window : index + window + horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


if __name__ == '__main__':
    # 1.加载数据文件
    data = load_st_dataset(dataset='PEMSD4', input_dim=1)
    print(data.shape)
    # 2.数据归一化
    data, scaler = normalize_dataset(data=data, normalizer="std")
    # 3.划分数据集
    train_data, val_data, test_data = split_data_by_ratio(data=data, val_ratio=0.2, test_ratio=0.2)
    # 4.滑动窗口采样
    x_tra, y_tra = Add_Window_Horizon(data=train_data, window=12, horizon=12, single=False)
    x_val, y_val = Add_Window_Horizon(data=val_data, window=12, horizon=12, single=False)
    x_test, y_test = Add_Window_Horizon(data=test_data, window=12, horizon=12, single=False)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)