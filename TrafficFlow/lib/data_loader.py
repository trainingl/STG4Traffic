import torch
import torch.utils.data
import warnings
from lib.generate_data import *
warnings.filterwarnings('ignore')

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
    data, scaler = normalize_dataset(data, normalizer)
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


if __name__ == '__main__':
    import argparse
    DATASET = 'PEMSD4'
    NODE_NUM = 307
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--input_dim', default=1, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--window', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, 
                                                                               normalizer='std', 
                                                                               tod=False,
                                                                               dow=False, 
                                                                               single=False)