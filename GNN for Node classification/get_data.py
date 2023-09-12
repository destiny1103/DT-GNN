# -*- coding:utf-8 -*-
import torch
import pandas as pd
import numpy as np
from scipy import sparse
from torch_geometric.data import Data
from util import device

# Load training data
def load_data(feature_path, graph_path, randomized_sampling=False, **kwargs):
    df = pd.read_csv(feature_path)
    num_in_feats = 10  # Number of features used for training
    num_out_feats = 3  # Number of output categories (N categories)
    # All features of this study ['LAT', 'LON', 'SOG', 'angle_COG', 'length', 'time_difference', 'velocity(m/s)', 'count', 'N1_Area', 'V_Area']
    node_features = df[['LAT', 'LON', 'SOG', 'angle_COG', 'length', 'time_difference', 'velocity(m/s)', 'count', 'N1_Area', 'V_Area']]  # Select Features to Use
    node_features = node_features.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # Feature normalization processing (necessary)

    df['Label'] = df['Label'].apply(lambda x: 2 if x == 5 else x)
    node_labels = df['Label'].tolist()
    graph = sparse.load_npz(graph_path)  # Read graph adjacent sparse matrix
    edge_weight = np.array(graph.data)
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
    edge_index, edge_weight = torch.from_numpy(np.vstack((graph.nonzero()[0], graph.nonzero()[1]))).long(), torch.from_numpy(edge_weight).to(torch.float)

    x = torch.tensor(node_features.values, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long).squeeze()
    if not randomized_sampling:
        N = len(node_features)
        # Assign the first n samples to the training set
        n = kwargs['sample_nums']
        train_mask = np.zeros(N, dtype=bool)
        train_mask[:n] = True
        # Randomly allocate the remaining samples to the validation and testing sets
        val_test_mask = np.zeros(N, dtype=int)
        val_test_mask[n:] = np.random.choice([1, 2], size=N - n, p=[0.2, 0.8])
        # Generate Verification Set Mask
        val_mask = np.zeros(N, dtype=bool)
        val_mask[val_test_mask == 1] = True
        # Generate Test Set Mask
        test_mask = np.zeros(N, dtype=bool)
        test_mask[val_test_mask == 2] = True
        dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, train_mask=train_mask, val_mask=val_mask,
                       test_mask=test_mask)
    else:
        n = len(node_features)
        mask = np.random.choice([0, 1, 2], size=n, p=[0.1, 0.1, 0.8])  # Proportion of training, validation, and test sets

        train_mask = np.zeros(n, dtype=bool)
        train_mask[mask == 0] = True
        val_mask = np.zeros(n, dtype=bool)
        val_mask[mask == 1] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[mask == 2] = True
        dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y, train_mask=train_mask, val_mask=val_mask,
                       test_mask=test_mask)

    return dataset, num_in_feats, num_out_feats


if __name__ == "__main__":
    feature_path = ".\\data\\NewYork\\Trajectory_label&features.csv"
    graph_path = ".\\data\\NewYork\\Graph_sparse_matrix.npz"
    dataset, num_in_feats, num_out_feats = load_data(feature_path, graph_path, sample_nums=3796, randomized_sampling=True)
    # print(dataset.edge_index)
    # print(dataset.edge_attr)
