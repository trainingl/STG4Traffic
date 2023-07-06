import os
import torch
import pickle
import numpy as np


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
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
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
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
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                yield (x_i, y_i, i_i)
                self.current_ind += 1
        return _wrapper()

    
class StandardScaler():
    """
    Standard the input
    """
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
        x_t = data[t + x_offsets, ...]   # (12, num_nodes, 2)
        y_t = data[t + y_offsets, ...]   # (12, num_nodes, 2)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0) # x, y: (samples, 12, num_nodes, 2)
    return x, y


def load_dataset(args, batch_size, valid_batch_size=None, test_batch_size=None, days=288):
    data = load_st_dataset(args.dataset, args.input_dim)
    x_offsets = np.arange(-(args.window - 1), 1, 1)  # array([-11, -10, ..., 0])
    y_offsets = np.arange(1, args.window + 1, 1)   # array([1, 2, ..., 12])
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)
    # Write the data into npz file.
    # train/val/test: 6 : 2 : 2
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)
    num_test = num_samples - num_train - num_val

    # train data
    x_train, y_train = x[:num_train], y[:num_train]
    # valid data
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test data
    x_test, y_test = x[-num_test:], y[-num_test:]
    data = {}
    for category in ['train', 'val', 'test']:
        data['x_' + category] = locals()["x_" + category][:, :, :, 0:2]  # B T N F speed flow
        data['y_' + category] = locals()['y_' + category][:, :, :, 0:1]
        if category == "train":
            data['scaler'] = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    for si in range(0, data['x_' + category].shape[-1]):
        scaler_tmp = StandardScaler(mean=data['x_train'][..., si].mean(), std=data['x_train'][..., si].std())
        for category in ['train', 'val', 'test']:
            # Only the labels are standardized
            data['x_' + category][..., si] = scaler_tmp.transform(data['x_' + category][..., si])
    
    print("train data", data['y_train'].shape, "val data", data['y_val'].shape, "test data", data['y_test'].shape)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                    begin=data['x_train'].shape[0])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                     begin=data['x_train'].shape[0] + data['x_val'].shape[0])
    return data


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
    return mae, mape, rmse


if __name__ == '__main__':
    import argparse
    DATASET = 'PEMSD4'
    NODE_NUM = 307
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--input_dim', default=2, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--window', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    data = load_dataset(
        args, 
        batch_size=args.batch_size, 
        valid_batch_size=args.batch_size, 
        test_batch_size=args.batch_size, 
        days=288
    )
    for idx, (x, y, ind) in enumerate(data['train_loader'].get_iterator()):
        print(x.shape, y.shape, ind.shape)
        break