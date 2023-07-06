# import sys
# sys.path.append('../')

import numpy as np
from scipy.sparse.linalg import eigs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lib.generate_data import load_st_dataset, StandardScaler


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict  # wd: this could overlap with 'label_start_index', e.g. when num_for_predict is larger than 12 (one hour)
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    if len(x_idx) != num_of_batches:
        return None
    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    week_indices = search_data(data_sequence.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7 * 24, points_per_hour)
    if not week_indices:
        return None
    day_indices = search_data(data_sequence.shape[0], num_of_days, label_start_idx, num_for_predict, 24, points_per_hour)
    if not day_indices:
        return None
    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour)
    if not hour_indices:
        return None
    week_sample = np.concatenate([data_sequence[i: j] for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j] for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    return week_sample, day_sample, hour_sample, target


class DatasetPEMS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        week_sample = self.data['week'][index]
        day_sample = self.data['day'][index]
        hour_sample = self.data['hour'][index]
        label = self.data['target'][index]

        return week_sample, day_sample, hour_sample, label


def read_and_generate_dataset(dataset, num_of_features, num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict=12,
                              points_per_hour=12, merge=False):
    # 1.加载数据集
    data_seq = load_st_dataset(dataset, input_dim=num_of_features)   # (T, N, D)
    # 2.特征标准化
    if num_of_features == 1:
        scaler = StandardScaler(mean=data_seq.mean(), std=data_seq.std())
        data_seq = scaler.transform(data_seq)
    else:
        scaler = StandardScaler(mean=data_seq[..., 0].mean(), std=data_seq[..., 0].std())
        data_seq[..., 0] = scaler.transform(data_seq[..., 0])
        for i in range(1, num_of_features):
            Nscaler = StandardScaler(mean=data_seq[..., i].mean(), std=data_seq[..., i].std())
            data_seq[..., i] = Nscaler.transform(data_seq[..., i])
    print("mean: ", scaler.mean, "std: ", scaler.std)
    # 3.按照 week、day、hours 三个范围进行采样
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours, idx, num_for_predict, points_per_hour)
        if not sample:
            continue
        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(day_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(hour_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(target, axis=0).transpose(0, 2, 1, 3)[:, :, :, 0:1]  # target feature is the traffic flow
        ))
    # 4.数据集划分
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    if not merge:
        training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set
    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(train_week.shape, train_day.shape, train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(test_week.shape, test_day.shape, test_hour.shape, test_target.shape))
    # 5.返回的数据格式
    all_data = {
        'train': {
            'week': train_week,
            'day': train_day,
            'hour': train_hour,
            'target': train_target,
        },
        'val': {
            'week': val_week,
            'day': val_day,
            'hour': val_hour,
            'target': val_target
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'hour': test_hour,
            'target': test_target
        },
        'scaler': scaler
    }
    return all_data


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


if __name__ == "__main__":
    all_data = read_and_generate_dataset(
        dataset="PEMSD8", 
        num_of_features=3, 
        num_of_weeks=2, 
        num_of_days=1,
        num_of_hours=2, 
        num_for_predict=12,
        points_per_hour=12,
        merge=0
    )
    # 1.training set data loader
    train_dataset = DatasetPEMS(all_data['train'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 2.validation set data loader
    val_dataset = DatasetPEMS(all_data['val'])
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # 3.testing set data loader
    test_dataset = DatasetPEMS(all_data['test'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for i, [train_w, train_d, train_h, label] in enumerate(train_loader):
        # train_w = train_w.to(device)
        # train_d = train_d.to(device)
        # train_r = train_r.to(device)
        # label = label.to(device)
        print(train_w.shape, train_d.shape, train_h.shape, label.shape)
        break

    adj_mx, distance_mx = get_adjacency_matrix("../data/PEMSD4/PEMSD4.csv", num_of_vertices=307, id_filename=None)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [i for i in cheb_polynomial(L_tilde, K=3)]
    print("cheb_polynomials: ", len(cheb_polynomials))