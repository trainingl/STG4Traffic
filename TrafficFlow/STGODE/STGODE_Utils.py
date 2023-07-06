import os
import csv
import torch
import numpy as np
from fastdtw import fastdtw
from lib.generate_data import load_st_dataset

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw
    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix
    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.dataset.lower()
    data = load_st_dataset(args.dataset)  # feature: flow, occupy, speed
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    if not os.path.exists(f'../data/{args.dataset}/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'../data/{args.dataset}/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'../data/{args.dataset}/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1

    # # use continuous semantic matrix
    # if not os.path.exists(f'data/{filename}_dtw_c_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')
    #     # normalization
    #     std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    #     mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    #     dist_matrix = (dist_matrix - mean) / std
    #     sigma = 0.1
    #     dtw_matrix = np.exp(- dist_matrix**2 / sigma**2)
    #     dtw_matrix[dtw_matrix < 0.5] = 0 
    #     np.save(f'data/{filename}_dtw_c_matrix.npy', dtw_matrix)
    # dtw_matrix = np.load(f'data/{filename}_dtw_c_matrix.npy')
    
    # use continuous spatial matrix
    if not os.path.exists(f'../data/{args.dataset}/{filename}_spatial_distance.npy'):
        with open(f'../data/{args.dataset}/{args.dataset}.csv', 'r') as fp:
            dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
            np.save(f'../data/{args.dataset}/{filename}_spatial_distance.npy', dist_matrix)

    # use 0/1 spatial matrix
    # if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
    #     dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
    #     sp_matrix = np.zeros((num_node, num_node))
    #     sp_matrix[dist_matrix != np.float('inf')] = 1
    #     np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    dist_matrix = np.load(f'../data/{args.dataset}/{filename}_spatial_distance.npy')
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres2] = 0 
    # np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
    return dtw_matrix, sp_matrix


def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


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


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer='std', tod=False, dow=False, single=False):
    # 1.加载数据集
    data = load_st_dataset(args.dataset, args.input_dim)
    # 2.数据归一化处理
    num_node = data.shape[1]
    means = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    stds = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - means) / stds
    mean_value = means.reshape(-1)[0]
    std_value = stds.reshape(-1)[0]
    print(data.shape, mean_value, std_value)
    scaler = StandardScaler(mean=mean_value, std=std_value)

    # 3.数据集划分(训练集、验证集、测试集)
    if args.test_ratio > 1:
        train_data, val_data, test_data = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        train_data, val_data, test_data = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # 4.滑动窗口采样
    x_tra, y_tra = Add_Window_Horizon(train_data, args.window, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(val_data, args.window, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(test_data, args.window, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    # 5.生成数据迭代器dataloader，并对训练集样本进行打乱
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_tra) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler