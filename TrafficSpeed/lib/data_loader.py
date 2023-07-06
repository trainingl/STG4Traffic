import os
import copy
import numpy as np


# 数据标准化
class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        data = (data * self.std) + self.mean
        return data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
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
        # Disrupt the original sample.
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


class DataLoaderM(object):
    def __init__(self, xs, ys, ycl, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            # ycl_padding = np.repeat(ycl[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ycl = np.concatenate([ycl, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ycl = ycl

    def shuffle(self):
        # Disrupt the original sample.
        permutation = np.random.permutation(self.size)
        xs, ys, ycl = self.xs[permutation], self.ys[permutation], self.ycl[permutation]
        self.xs = xs
        self.ys = ys
        self.ycl = ycl

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                ycl_i = self.ycl[start_ind: end_ind, ...]
                yield (x_i, y_i, ycl_i)
                self.current_ind += 1
        return _wrapper()


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    # Load data
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']   # (, timestep, num_node, feature_dim)
        data['y_' + category] = cat_data['y']   # (, timestep, num_node, feature_dim)
    # Data format
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        # 注意：这里同时对 x_train、x_val、x_test 进行了归一化，对于标签 y 并没有做归一化
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    print("train:", data['x_train'].shape, " val:", data['x_val'].shape, " test:", data['x_test'].shape)
    # 用于监督训练 teacher forcing
    data['ycl_train'] = copy.deepcopy(data['y_train'])
    data['ycl_train'][..., 0] = scaler.transform(data['y_train'][..., 0])
    
    # Iterator to initialize the dataset
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], data['ycl_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


# 测试上面的方法
# dataset_dir = '../data/METR-LA/processed/'
# batch_size = 64
# data = load_dataset(dataset_dir, batch_size, valid_batch_size=batch_size, test_batch_size=batch_size)