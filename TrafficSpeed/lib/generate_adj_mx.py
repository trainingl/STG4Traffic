import pickle
import argparse
import numpy as np
import pandas as pd

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: dataframe with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # build sensor id to index map
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        # directed graph
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # calculate the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    # print(len(distances))
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))  # RBF
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # set entries that lower than a threshold to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


# if __name__ == '__main__':
#     # 方法的参数信息
#     parser = argparse.ArgumentParser()
#     """
#     PEMS-BAY:
#         parser.add_argument("--sensor_locations_filename", type=str, default="../data/PEMS-BAY/graph_sensor_locations_bay.csv")
#         parser.add_argument("--distances_filename", type=str, default="../data/PEMS-BAY/distances_bay_2017.csv")
#         parser.add_argument("--output_pkl_filname", type=str, default="../data/PEMS-BAY/processed/adj_mx.pkl")
#     """
#     # METR-LA
#     parser.add_argument("--sensor_locations_filename", type=str, default="../data/METR-LA/graph_sensor_locations.csv")
#     parser.add_argument("--distances_filename", type=str, default="../data/METR-LA/distances_la_2012.csv")
#     parser.add_argument("--normalized_k", type=float, default=0.1)
#     parser.add_argument("--output_pkl_filname", type=str, default="../data/METR-LA/processed/adj_mx.pkl")
#     args = parser.parse_args()
    
#     location_df = pd.read_csv(args.sensor_locations_filename, dtype={'sensor_id': 'str'})
#     sensor_ids = list(location_df['sensor_id'])
#     distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
#     normalized_k = args.normalized_k
#     _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
#     print(adj_mx[:5, :5])
    # save to pickle file.
    # with open(args.output_pkl_filname, 'wb') as f:
    #     # Serialize the object into the opened file.
    #     pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)