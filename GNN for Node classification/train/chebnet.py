import os
import pandas as pd
from get_data import load_data
from models import ChebNet
from util import device, train

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # load data (Choose NewYork or Singapore dataset)
    feature_path = "..\\data\\Singapore\\Trajectory_label&features.csv"
    graph_path = "..\\data\\Singapore\\Graph_sparse_matrix.npz"
    dataset, num_in_feats, num_out_feats = load_data(feature_path, graph_path, sample_nums=3796, randomized_sampling=True)
    # model training
    k = 1  # Chebyshev polynomials of order (default: 1)
    model = ChebNet(num_in_feats, 64, num_out_feats, k).to(device)
    model.set_use_edge_weight(False)  # using  model with edge weights (True or False)
    model, test_acc, bestpred, train_time, lossandacc = train(model, dataset)
    print('test acc:', test_acc, 'train time:', train_time)
    # output results
    lossandacc = pd.DataFrame(lossandacc)
    lossandacc.to_csv("..\\output\\Singapore\\ChebNet-lossandacc1.csv", header=False, index=False)
    bestpred = pd.DataFrame(bestpred.cpu().numpy())
    bestpred.to_csv("..\\output\\Singapore\\ChebNet-bestpred1.csv", header=False, index=False)


if __name__ == '__main__':
    main()
