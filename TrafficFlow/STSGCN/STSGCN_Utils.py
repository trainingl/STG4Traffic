import numpy as np
import torch
import os
import pandas as pd

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


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


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    num_samples = data.shape[0]
    x, y = [], []
    min_t = abs(min(x_offsets))    # 11
    max_t = abs(num_samples - max(y_offsets))   # seq_length - 12
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]   # (12, num_nodes, 1)
        y_t = data[t + y_offsets, ...]   # (12, num_nodes, 1)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0) # x, y: (samples, 12, num_nodes, 1)
    return x, y


def load_dataset(args, batch_size, valid_batch_size=None, test_batch_size=None):
    data = load_st_dataset(args.dataset, args.input_dim)
    x_offsets = np.arange(-(args.window - 1), 1, 1)  # array([-11, -10, ..., 0])
    y_offsets = np.arange(1, args.window + 1, 1)   # array([1, 2, ..., 12])
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)
    # Write the data into npz file.
    # train/val/test: 6 : 2 : 2
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.6)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_train - num_test

    # train data
    x_train, y_train = x[:num_train], y[:num_train]
    # valid data
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test data
    x_test, y_test = x[-num_test:], y[-num_test:]
    data = {}
    for category in ['train', 'val', 'test']:
        data['x_' + category] = locals()["x_" + category][:, :, :, 0:1]
        data['y_' + category] = locals()['y_' + category][:, :, :, 0:1]
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler
    return data


def get_Gaussian_matrix(distance_df_filename, num_of_vertices, normalized_k=0.1, id_filename=None):
    num_sensors = num_of_vertices
    A = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # N * N
    A[:] = np.inf   # 初始化无穷大(表示无穷远)
    
    if id_filename:
        with open(id_filename, 'r') as f:
            # 建立索引表，注意分隔符
            sensor_ids = f.read().strip().split('\n')
            id_dict = {
                int(i): idx for idx, i in enumerate(sensor_ids)
            }
            df = pd.read_csv(distance_df_filename)
            for row in df.values:
                if row[0] not in id_dict or row[1] not in id_dict:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                # 无向图还是有向图
                A[id_dict[i], id_dict[j]] = distance
                A[id_dict[j], id_dict[i]] = distance
    else:
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            # 区分数据是有向图还是无向图
            A[i, j] = distance
            A[j, i] = distance
    distances = A[~np.isinf(A)].flatten()
    std = distances.std()   # 计算距离的方差
    adj_mx = np.exp(-np.square(A / std))
    adj_mx[adj_mx < normalized_k] = 0   # 系数化处理
    # 返回预处理好的邻接矩阵
    return adj_mx

# Localized Spatial-Temporal Graph Construction
def construct_adj(A, steps):
    """
    :params A: Binary matrix, shape is (N, N).
    :params steps: Select time steps to build a local space-time graph, generally 3.
    :return: Localized spatial-temporal graph, shape is (N * steps, N * steps).
    """
    N = len(A)  # Get the number of rows
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        # The diagonal represents the spatial-temporal graph of each timestep.
        adj[i * N : (i + 1) * N, i * N : (i + 1) * N] = A
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1 
    for i in range(len(adj)):
        # add self-loop
        adj[i, i] = 1
    return adj

##################################################################################

def huber_loss(pred, labels,rho=1,null_val=np.nan): # loss
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(pred - labels)
    loss = torch.where(loss > rho, loss - 0.5 * rho, (0.5 / rho) * torch.square(loss))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, rmse, mape
