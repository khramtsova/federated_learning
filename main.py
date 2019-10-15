import time
import copy
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from data.get_data import get_dataloaders
from model.model import Net
from model.train import train
from model.aggregation import FedAvg
from data.data_distribution import Distribute

batch_size = 200
num_workers = 2
classes = [i for i in range(10)]


def main():

    # Initialize the visualization environment
    accuracy = []
    loss_train_global, loss_test_global = [], []
    loss_train_local, loss_test_local = [], []

    distrib = Distribute(num_workers, len(classes))

    train_data_distribution = distrib.create_non_iid_exclusive(2)
    test_data_distribution = distrib.create_iid()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainloaders, testloaders = get_dataloaders(train_data_distribution,
                                                test_data_distribution,
                                                "cifar",
                                                20,
                                                2)

    net = Net(10)

    # copy weights
    # w_glob = net.state_dict()

    net.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):

        net.train()

        w_local, loss_local = [], []

        # For now all of the updates are calculated sequentially
        for trainloader in trainloaders:
            w, loss = train(net, trainloader, loss_func, optimizer, 10, device=device)
            w_local.append(copy.deepcopy(w))
            loss_local.append(copy.deepcopy(loss))

        w_glob = FedAvg(w_local)

        # Analog to model distribution
        net.load_state_dict(w_glob)
        loss_avg = sum(loss_local) / len(loss_local)

        # print loss
        print(loss_avg, loss_local)

        loss_train_global.append(loss_avg)
        loss_train_local.append(loss_local)

    print("End of training")

    # ====================== PLOT ==========================


    loss_train_local = np.transpose(loss_train_local)
    for i,l in enumerate(loss_train_local):
        plt.plot(l, label='Agent '+i)
    plt.plot(loss_train_local, label='Average loss')
    plt.legend(frameon=False)
    plt.savefig("./logs/loss.png")

    # ====================== PLOT ==========================

    print("Saving model to ./logs")
    adrs = "./logs/model" + "".join(map(str, classes)) +".pth"
    torch.save(net.state_dict(), adrs)

    """
    plt.plot(accuracy, label='Validation accuracy')
    plt.legend(frameon=False)
    plt.savefig("./logs/accuracy.png")
    plt.clf()
    """

if __name__ == '__main__':
    main()
