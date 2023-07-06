# import sys
# sys.path.append('../')

import torch
from lib.generate_data import *
from model.STGNCDE.controldiffeq.interpolate import natural_cubic_spline_coeffs

def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)
    # X = tuple(TensorFloat(x) for x in X)
    # Y = TensorFloat(Y)
    data = torch.utils.data.TensorDataset(*X, torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader_cde(args, normalizer = 'std', tod=False, dow=False, single=True):
    # load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    # normalize stdata
    data, scaler = normalize_dataset(data, normalizer)
    # spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.window, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.window, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.window, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    # TODO: make argument for missing data
    if args.missing_test == True:
        generator = torch.Generator().manual_seed(56789)
        xs = np.concatenate([x_tra, x_val, x_test])
        for xi in xs:
            removed_points_seq = torch.randperm(xs.shape[1], generator=generator)[:int(xs.shape[1] * args.missing_rate)].sort().values
            removed_points_node = torch.randperm(xs.shape[2], generator=generator)[:int(xs.shape[2] * args.missing_rate)].sort().values
            for seq in removed_points_seq:
                for node in removed_points_node:
                    xi[seq,node] = float('nan')
        x_tra = xs[:x_tra.shape[0],...] 
        x_val = xs[x_tra.shape[0]:x_tra.shape[0]+x_val.shape[0],...]
        x_test = xs[-x_test.shape[0]:,...] 
    
    # TODO: make argument for data category
    data_category = 'traffic'
    if data_category == 'traffic':
        times = torch.linspace(0, 11, 12)
    elif data_category == 'token':
        times = torch.linspace(0, 6, 7)
    else:
        raise ValueError
    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

    train_coeffs = natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = natural_cubic_spline_coeffs(times, x_test.transpose(1,2))
    
    ############## get dataloader ######################
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, times


if __name__ == '__main__':
    import argparse
    DATASET = 'PEMSD4'
    NODE_NUM = 307
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--input_dim', default=1, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--window', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--missing_test', default=False, type=bool)
    parser.add_argument('--missing_rate', default=0.1, type=float)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler, times = get_dataloader_cde(args, 
                                                                                normalizer='std', 
                                                                                tod=False,
                                                                                dow=False, 
                                                                                single=False)
    for batch_idx, batch in enumerate(train_dataloader):
        batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
        *valid_coeffs, target = batch  # list: 4, tensor: (64, 12, 307, 1)
        print(len(valid_coeffs), target.shape)
        break