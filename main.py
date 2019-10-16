import time
import copy
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from data.get_data import get_dataloaders
from model.model import Net
from model.train import train, test
from model.aggregation import FedAvg
from data.data_distribution import Distribute

batch_size = 32
num_workers = 5
classes = [i for i in range(10)]
rounds = 2
# For non-iid
n_class_per_user = 2


def main():

    # Initialize the visualization environment

    acc_test, loss_test, acc_train, loss_train = [], [], [], []
    avrg = lambda a: sum(a)/len(a)

    distrib = Distribute(num_workers, len(classes))

    train_data_distribution = distrib.create_iid()
    test_data_distribution = distrib.create_iid()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainloaders, testloaders = get_dataloaders(train_data_distribution,
                                                test_data_distribution,
                                                "cifar",
                                                batch_size,
                                                num_workers)

    net = Net(10)

    # copy weights
    # w_glob = net.state_dict()

    net.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for rnd in range(rounds):

        net.train()

        w_local = []

        # For now all of the updates are calculated sequentially
        for trainloader in trainloaders:
            w, loss, acc = train(net, trainloader, loss_func, optimizer, 2, device=device)

            w_local.append(copy.deepcopy(w))
            loss_train.append(copy.deepcopy(loss))
            acc_train.append(copy.deepcopy(acc))

        w_glob = FedAvg(w_local)

        # Analog to model distribution
        net.load_state_dict(w_glob)

        # Perform tests after global update

        for testloader in testloaders:
            loss, acc = test(net, testloader, loss_func, device=device)
            acc_test.append(copy.deepcopy(acc))
            loss_test.append(copy.deepcopy(loss))

        print("Round", rnd)
    print("End of training")

    acc_train = np.reshape(acc_train, [rounds, num_workers])
    loss_train = np.reshape(loss_train, [rounds, num_workers])
    acc_test = np.reshape(acc_test, [rounds, num_workers])
    loss_test = np.reshape(loss_test, [rounds, num_workers])

    print(acc_train, "\n\n", loss_train)

    # ====================== PLOT ==========================

    for i,l in enumerate(loss_train.T):
        plt.plot(l, label='Agent ' + str(i))
    #plt.plot(loss_train_local, label='Average loss')
    plt.legend(frameon=False)
    plt.title("Train loss")
    plt.savefig("./logs/loss_train.png")
    plt.clf()

    for i,l in enumerate(loss_test.T):
        plt.plot(l, label='Agent ' + str(i))
    #plt.plot(loss_train_local, label='Average loss')
    plt.legend(frameon=False)
    plt.title("Test loss")
    plt.savefig("./logs/loss_test.png")
    plt.clf()

    for i,l in enumerate(acc_train.T):
        plt.plot(l, label='Agent '+ str(i))
    plt.legend(frameon=False)
    plt.title("Train accuracy")
    plt.savefig("./logs/acc_train.png")
    plt.clf()

    for i,l in enumerate(acc_test.T):
        plt.plot(l, label='Agent '+ str(i))
    #plt.plot(loss_train_local, label='Average loss')
    plt.legend(frameon=False)
    plt.title("Test accuracy")
    plt.savefig("./logs/acc_test.png")
    plt.clf()

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
