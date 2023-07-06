import os
import argparse
import numpy as np
import pandas as pd

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
    num_samples, num_nodes = df.shape   # (seq_length, num_nodes)
    data = np.expand_dims(df.values, axis=-1)  # (seq_length, num_nodes, 1)
    data_list = [data]
    if add_time_in_day: # True
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) 
        # (1, 1, times) -> copy -> (1, num_nodes, times) -> transpose -> (seq_length, num_nodes, 1)
        data_list.append(time_in_day)
    if add_day_in_week: # False
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))    # (seq_length, num_nodes, 7)
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1)  # (seq_length, num_nodes, 2)
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


def generate_data_h5(args, input_dir, output_dir):
    """
       input_dir: 输入数据的文件路径 
       output_dir: 输出数据的文件路径
    """
    df = pd.read_hdf(input_dir)
    x_offsets = np.arange(-(args.window - 1), 1, 1)  # array([-11, -10, ..., 0])
    y_offsets = np.arange(1, args.horizon + 1, 1)   # array([1, 2, ..., 12])
    # x: (num_samples, seq_length, num_nodes, input_dim)
    # y: (num_samples, seq_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )
    print("x shape: ", x.shape, ", shape: ", y.shape)

    # Write the data into npz file.
    # train/val/test: 7 : 1 : 2
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_rate)
    num_val = round(num_samples * args.val_rate)
    num_test = num_samples - num_train - num_val

    # train data
    x_train, y_train = x[:num_train], y[:num_train]
    # valid data
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # test data
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()['y_' + cat]
        print(cat, 'x: ', _x.shape, "y: ", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),  # (12, 1)
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1])   # (12, 1)
        )


def main(args):
    print("Generating training data: ")
    if args.dataset == "METR-LA":
        print("METR-LA: ")
        generate_data_h5(args, "../data/METR-LA/metr-la.h5", "../data/METR-LA/processed/")
    elif args.dataset == "PEMS-BAY":
        print("PEMS-BAY: ")
        generate_data_h5(args, "../data/PEMS-BAY/pems-bay.h5", "../data/PEMS-BAY/processed/")
    elif args.dataset == "PEMS":
        print("PEMS: ")   # 扩展数据接口
    print("Finish!")


"""
    该方法仅对数据进行采样、按比例划分，并未对数据进行规范化操作
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train_rate", type=float, default=0.7)
    parser.add_argument("--val_rate", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="METR-LA")
    args = parser.parse_args()
    # bash: python generate_data.py --dataset METR-LA  /  python generate_data.py --dataset PEMS-BAY
    main(args)