import copy
import torch


def FedAvg(w):

    w_avg = copy.deepcopy(w[0])
    for j in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[j] += w[i][j]
        w_avg[j] = torch.div(w_avg[j], len(w))

    return w_avg