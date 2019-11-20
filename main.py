
import copy
import numpy as np

import torch
import torch.nn as nn
import syft as sy
from syft.frameworks.torch.federated.utils import extract_batches_per_worker, federated_avg

from data.get_data import get_dataloaders
from model.model import Net
from model.train import train, test
from model.aggregation import FedAvg
from data.data_distribution import Distribute
from src.log_saver import LogSaver
from src.options import args_parser


def main():

    args = args_parser()

    logs = LogSaver(args)

    acc_test, loss_test, acc_train, loss_train = [], [], [], []

    # ToDo change this
    classes = [i for i in range(args.num_classes)]

    distrib = Distribute(args.num_workers, len(classes))

    train_data_distribution = copy.deepcopy(distrib.get_distribution(args.dstr_Train,
                                                                     args.n_labels_per_agent_Train,
                                                                     args.sub_labels_Train))

    test_data_distribution = copy.deepcopy(distrib.get_distribution(args.dstr_Test,
                                                       args.n_labels_per_agent_Test,
                                                       args.sub_labels_Test))

    print(train_data_distribution, "\n\n TEST DISTRIBUTION", test_data_distribution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    fed_trainloaders, fed_testloaders, workers = get_dataloaders(train_data_distribution,
                                                 test_data_distribution,
                                                 args.dataset,
                                                 args.train_bs,
                                                 args.test_bs,
                                                 args.num_workers)
    print("TRAINLOADERs ARE CREATED")

    batches = extract_batches_per_worker(fed_trainloaders)
    batches_test = extract_batches_per_worker(fed_testloaders)

    net = Net(10)

    # copy weights
    # w_glob = net.state_dict()

    net.to(device)
    loss_func = nn.CrossEntropyLoss()

    for rnd in range(args.rounds):

        w_local = {}
        n = []

        # For now all of the updates are calculated sequentially
        for worker in workers:

            trainloader = batches[worker]

            # Batch size is needed to calculate accuracy
            w, loss, acc = train(worker, net, trainloader, loss_func, args.local_ep, args.train_bs, device=device)
            # ToDo w -> w.state_dict()
            w_local[worker] = w #.state_dict()
            n.append(len(trainloader))
            loss_train.append(copy.deepcopy(loss))
            acc_train.append(copy.deepcopy(acc))

        net = federated_avg(w_local)

        # w_glob = FedAvg(w_local, n)
        # Analog to model distribution
        # net.load_state_dict(w_glob)

        # Perform tests after global update
        for worker in workers:
            testloader = batches_test[worker]
            loss, acc = test(worker, net, testloader, loss_func, args.test_bs, device=device)

            print(worker.id, "loss", loss, "acc", acc)
            acc_test.append(copy.deepcopy(acc))
            loss_test.append(copy.deepcopy(loss))

        print("Round", rnd)
        #print(acc_train[-1], loss_train[-1], acc_test[-1], loss_test[-1])
        logs.add_row(acc_train, loss_train, acc_test, loss_test)

    print("End of training")

    print(acc_train, "\n\n", type(acc_train))

    logs.plot(loss_train, loss_test, np.array(acc_train), np.array(acc_test))
    print("Plots are created\n", acc_train, "\n\n", loss_train)
    logs.save_model(net)


if __name__ == '__main__':
    main()
    #LogSaver("sd")
