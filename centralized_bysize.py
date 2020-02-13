from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from syft.frameworks.torch.federated.utils import extract_batches_per_worker, federated_avg

from src.log_saver import LogSaver
from src.options import args_parser
from url_data.get_data import get_dataloaders
from model.model import LinearModel
from model.train import train, test


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        m.reset_parameters()



# Take equal amount of all the features apart
# All the workers have a same validation subset
def train_validation_split(frame, percent):
    label_count = frame.groupby("URL_Type_obf_Type")["Len_Query"].count()
    # amount = (label_count/4).astype(int)
    amount = (label_count * percent / 100).astype(int)
    amount["benign"] = 1
    # workers = dict.fromkeys([0,1,2,3], pd.DataFrame())

    # For each type of the attack
    test_data = pd.DataFrame()
    for name, subframe in frame.groupby("URL_Type_obf_Type"):
        test_data = test_data.append(subframe.sample(n=amount[name]))
    frame = frame.drop(test_data.index)
    # for worker_id in range(4):
    # Random sample of data

    # workers[worker_id] = workers[worker_id].append(temp)
    # subframe = subframe.drop(temp.index)
    return frame, test_data


# Separate target into a separate column and make it binary. Return Tensors
def data_target_split(dataset):

    target = dataset.pop("URL_Type_obf_Type")
    how_replace = dict(zip(["defacement", "benign", "malware", "phishing", "spam"],
                           [1, 0, 1, 1, 1]))
    target = target.map(how_replace)
    x = torch.from_numpy(dataset.values).float()
    y = torch.from_numpy(target.values)  # .long()

    return x, y


def get_data(args, train_data, test_data):

    x_train, y_train = data_target_split(train_data)
    x_test, y_test = data_target_split(test_data)
    train = TensorDataset(x_train, y_train)
    test = TensorDataset(x_test, y_test)

    #print(type(train[0]), type(train[1]),train[0].shape, train[1].shape)
    #print(test[0].shape, test[1].shape)

    train_loader = DataLoader(train, batch_size=args.train_bs, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.test_bs, shuffle=True)
    return train_loader, test_loader


def train_and_test(args, trainloaders, testloaders, net,
                   loss_func, device, worker=0):

    logs = LogSaver(args, logsubfolder=str(worker)+"_"+str(len(trainloaders)*args.train_bs))

    net.apply(weight_init)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    acc_test, loss_test, acc_train, loss_train = [], [], [], []
    avrg = lambda a: sum(a) / len(a)

    for rnd in range(args.rounds):

        net.train()
        batch_loss = []  # torch.Tensor(len(loader)).send(worker)
        correct = 0
        for indx, (images, labels) in enumerate(trainloaders):
            images, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        loss_train.append(copy.deepcopy(avrg(batch_loss)))
        correct_value = correct.float().item()
        acc_train.append(correct_value * 100. / (len(trainloaders) * args.train_bs))
        print(avrg(batch_loss), correct_value * 100. / (len(trainloaders) * args.train_bs))
        # TEST
        net.eval()
        test_loss = []
        correct = 0

        with torch.no_grad():
            for idx, (data, labels) in enumerate(testloaders):
                data, labels = data.to(device), labels.to(device)

                log_probs = net(data)
                loss = loss_func(log_probs, labels)
                test_loss.append(loss.item())

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

        correct_value = correct.float().item()
        accuracy = correct_value * 100. / (len(testloaders) * args.test_bs)
        acc_test.append(accuracy)
        loss_test.append(avrg(test_loss))
        logs.add_row(acc_train, loss_train, acc_test, loss_test)

    print(acc_test, "\n", acc_train)
    logs.plot(loss_train, loss_test, np.array(acc_train), np.array(acc_test))
    logs.save_model(net)


def main():

    args = args_parser()

    inputDim = 72
    outputDim = 2
    address_indx_train = args.logfolder + 'trainloader.pkl'
    address_indx_test = args.logfolder + 'testloader.pkl'

    dataset = pd.read_csv(args.data_folder, low_memory=False, na_values='NaN')
    with open(address_indx_train, 'rb') as output:
        indexes_train = pickle.load(output)

    with open(address_indx_test, 'rb') as output:
        indexes_test =  pickle.load(output)


    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = LinearModel(inputDim, outputDim)



    loss_func = nn.CrossEntropyLoss()

    net.to(device)

    for i in indexes_test:
        trainloaders, testloaders = get_data(args, dataset.loc[indexes_train[i]], dataset.loc[indexes_test[i]])
        train_and_test(args, trainloaders, testloaders, net, loss_func, device, len(indexes_train[i]))
        print("Done with worker", i)
    #epoch_loss = []
    #epoch_acc = []




if __name__ == '__main__':
    main()