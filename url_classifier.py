from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import torch
import torch.nn as nn
from syft.frameworks.torch.federated.utils import extract_batches_per_worker

from src.log_saver import LogSaver
from src.options import args_parser
from url_data.get_data import get_dataloaders
from model.model import LinearModel
from model.train import train, test
from model.aggregation import FedAvg


def main():

    inputDim = 72
    outputDim = 2

    args = args_parser()
    logs = LogSaver(args)

    fed_trainloaders, fed_testloaders, workers, worker_sizes = get_dataloaders(args.data_folder,
                                                                               logs,
                                                                               args.dstr_Train,
                                                                               args.dstr_Test,
                                                                               args.num_workers,
                                                                               args.train_bs, args.test_bs)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ToDo: inefficient batch extraction
    batches = extract_batches_per_worker(fed_trainloaders)
    batches_test = extract_batches_per_worker(fed_testloaders)

    net = LinearModel(inputDim, outputDim)
    net.to(device)

    # If iid - compare to a centralized
    if args.dstr_Train == "iid":
        net1 = LinearModel(inputDim, outputDim)
        net1.to(device)
        logs.args.num_workers += 1


    # copy weights
    # w_glob = net.state_dict()


    loss_func = nn.CrossEntropyLoss()

    acc_test, loss_test, acc_train, loss_train = [], [], [], []
    for rnd in range(args.rounds):

        w_local = {}
        #n = []

        # For now all of the updates are calculated sequentially
        for worker in workers:
            trainloader = batches[worker]

            # Batch size is needed to calculate accuracy
            w, loss, acc = train(worker, net, trainloader, loss_func, args.local_ep, args.train_bs,
                                 device=device,
                                 lr=args.lr,
                                 optim=args.optimizer)

            w_local[worker] = w.state_dict()

            loss_train.append(copy.deepcopy(loss))
            acc_train.append(copy.deepcopy(acc))

        w_glob = FedAvg(list(w_local.values()), worker_sizes)
        # Analog to model distribution
        net.load_state_dict(w_glob)

        #net = federated_avg(w_local)

        # Perform tests after global update
        # If the test subset is the same for everyone - perform only one test
        if args.dstr_Test == "same":
            testloader = batches_test[worker]

            loss, acc = test(worker, net, testloader, loss_func, args.test_bs, device=device)

            acc_test.append(copy.deepcopy(acc))
            loss_test.append(copy.deepcopy(loss))
        else:
            for worker in workers:
                testloader = batches_test[worker]

                loss, acc = test(worker, net, testloader, loss_func, args.test_bs, device=device)

                acc_test.append(copy.deepcopy(acc))
                loss_test.append(copy.deepcopy(loss))

        # If iid - take last worker and perform a centralized training
        if args.dstr_Train == "iid":
            centr_worker = workers[0]
            print("worker", centr_worker)
            centr_train_loader = batches[centr_worker]
            net1, loss, acc = train(centr_worker, net1, centr_train_loader, loss_func, args.local_ep, args.train_bs, device=device)
            print("Centralized train loss", loss)
            loss_train.append(copy.deepcopy(loss))
            acc_train.append(copy.deepcopy(acc))
            centr_test_loader = batches_test[centr_worker]
            loss, acc = test(centr_worker, net1, centr_test_loader, loss_func, args.test_bs, device=device)
            acc_test.append(copy.deepcopy(acc))
            loss_test.append(copy.deepcopy(loss))
            print("Centralized test loss", loss)

        print("Round", rnd)
        logs.add_row(acc_train, loss_train, acc_test, loss_test)

    print("End of training")
    print(acc_train, "\n\n", type(acc_train))
    print("Plots are created\n", acc_train, "\n\n", loss_train)

    logs.plot(loss_train, loss_test, np.array(acc_train), np.array(acc_test))
    logs.save_model(net)


if __name__ == '__main__':
    main()