
import copy

import torch
import torch.optim as optim
import torch.nn as nn

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

    trainloaders, testloaders = get_dataloaders(train_data_distribution,
                                                test_data_distribution,
                                                args.dataset,
                                                args.train_bs,
                                                args.test_bs,
                                                args.num_workers)

    net = Net(10)

    # copy weights
    # w_glob = net.state_dict()

    net.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for rnd in range(args.rounds):

        net.train()

        w_local = []

        # For now all of the updates are calculated sequentially
        for trainloader in trainloaders:
            w, loss, acc = train(net, trainloader, loss_func, optimizer, args.local_ep, device=device)

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

        logs.add_row(acc_train, loss_train, acc_test, loss_test)

        print("Round", rnd)
    print("End of training")

    logs.plot(loss_train,loss_test, acc_train, acc_test)
    print("Plots are created\n", acc_train, "\n\n", loss_train)
    logs.save_model(net)


if __name__ == '__main__':
    main()
    #LogSaver("sd")
