import os
import pandas as pd
from get_data import load_data
from models import GraphSAGE
from util import train, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # load data (Choose NewYork or Singapore dataset)
    feature_path = "..\\data\\Singapore\\Trajectory_label&features.csv"
    graph_path = "..\\data\\Singapore\\Graph_sparse_matrix.npz"
    dataset, num_in_feats, num_out_feats = load_data(feature_path, graph_path, sample_nums=3796, randomized_sampling=True)
    model = GraphSAGE(num_in_feats, 64, num_out_feats).to(device)
    model, test_acc, bestpred, train_time, lossandacc = train(model, dataset)
    print('test acc:', test_acc, 'train time:', train_time)

    lossandacc = pd.DataFrame(lossandacc)
    lossandacc.to_csv("..\\output\\Singapore\\Graphsage-lossandacc.csv", header=False, index=False)
    bestpred = pd.DataFrame(bestpred.cpu().numpy())
    bestpred.to_csv("..\\output\\Singapore\\Graphsage-bestpred.csv", header=False, index=False)


if __name__ == '__main__':
    main()
