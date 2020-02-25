
import os
import csv
import math
import random
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch

from src.utils import get_project_root


class LogSaver:

    def __init__(self, args, logsubfolder=None):

        self.args = args

        # ToDo change the name of the folder, random for now
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

        # Create a new folder for every experiment
        if logsubfolder:
            self.folder = str(args.log_folder + "/" + logsubfolder + "/")

        else:
            self.folder = str(args.log_folder + random_str + "/")

        os.mkdir(self.folder)

        # Write summary to a file
        f1 = open(self.folder + "description.txt", 'w')
        f1.write(str(vars(args)))
        f1.close()

        file = open(self.folder + "result.csv", 'w')
        fieldnames = ['acc_train', 'loss_train', 'acc_test', 'loss_test']
        self.writer = csv.DictWriter(file, fieldnames=fieldnames)

    def add_row(self, acc_train, loss_train, acc_test, loss_test):
        self.writer.writerow({'acc_train': acc_train[-self.args.num_workers:],
                              'loss_train': loss_train[-self.args.num_workers:],
                              'acc_test': acc_test[-self.args.num_workers:],
                              'loss_test': loss_test[-self.args.num_workers:]})

    def plot(self, loss_train, loss_test, acc_train, acc_test):

        acc_train = np.reshape(acc_train, [self.args.rounds,  self.args.num_workers])
        loss_train = np.reshape(loss_train, [self.args.rounds, self.args.num_workers])
        #acc_test = np.reshape(acc_test, [self.args.rounds, self.args.num_workers])
        #loss_test = np.reshape(loss_test, [self.args.rounds, self.args.num_workers])

        acc_test = np.reshape(acc_test, [self.args.rounds, -1])
        loss_test = np.reshape(loss_test, [self.args.rounds, -1])

        # ====================== PLOT ==========================

        fig, axs = plt.subplots(2, 2, figsize=(20, 15))

        for i, l in enumerate(loss_train.T):
            axs[0,0].plot(l, label='Agent ' + str(i))
        axs[0,0].legend(frameon=False)
        axs[0,0].set_title("Train loss")

        for i, l in enumerate(loss_test.T):
            axs[0, 1].plot(l, label='Agent ' + str(i))
        axs[0,1].legend(frameon=False)
        axs[0,1].set_title("Test loss")

        for i, l in enumerate(acc_train.T):
            axs[1,0].plot(l, label='Agent ' + str(i))
        axs[1,0].legend(frameon=False)
        axs[1,0].set_title("Train accuracy")

        for i, l in enumerate(acc_test.T):
            axs[1,1].plot(l, label='Agent ' + str(i))
        axs[1,1].legend(frameon=False)
        axs[1,1].set_title("Test accuracy")

        fig.suptitle("Results", fontsize=16)
        plt.savefig(self.folder + "results.png")
        plt.clf()

        # ====================== PLOT ==========================

    # Distribution: dict(pd.Series);
    # len(dict) == num_workers
    def plot_distribution(self, distribution, filename):
        # If the distribution is different for each worker,
        # The class distribution is a dictionary of Series
        if isinstance(distribution, dict):
            max_columns = 3
            max_rows = math.ceil(self.args.num_workers / max_columns)
            fig, axs = plt.subplots(max_rows, max_columns, figsize=(22,10))
            for i in range(max_rows):
                for j in range(max_columns):
                    worker_id = i * max_columns + j
                    if worker_id != self.args.num_workers:
                        axs[i, j].bar("benign", distribution[worker_id]["benign"])
                        distribution[worker_id].drop(labels=["benign"], inplace=True)
                        malicious = sum(distribution[worker_id])
                        axs[i, j].bar(distribution[worker_id].index, distribution[worker_id].values)
                        axs[i, j].bar("Malicious", malicious)
                    else:
                        fig.suptitle("Data distribution", fontsize=16)
                        plt.savefig(self.folder + filename + ".png")
                        plt.clf()
                        break

        # Otherwise, all the users have the same label distribution and
        # we only make one plot
        # => The class distribution is one Series
        else:
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar("benign", distribution["benign"])
            distribution.drop(labels=["benign"], inplace=True)
            malicious = sum(distribution)
            ax.bar(distribution.index, distribution.values)
            ax.bar("Malicious", malicious)
            ax.set_title("Data distribution",  fontsize=16, fontweight='bold')
            plt.savefig(self.folder + filename +".png")
            plt.clf()
        return True

    def save_loaders(self, trainloader, testloader):

        print("Saving loaders")
        indx ={}
        for i in trainloader:
            indx[i] = trainloader[i].index
        with open(self.folder +'trainloader.pkl', 'wb') as output:
            pickle.dump(indx, output)

        indx = {}
        for i in testloader:
            indx[i] = testloader[i].index
        with open(self.folder +'testloader.pkl', 'wb') as output:
            pickle.dump(indx, output)

    def save_model(self, net, name=None):
        print("Saving model weights ")
        if name:
            adrs = self.folder + name + ".pth"
        else:
            adrs = self.folder + "model" + ".pth"
        torch.save(net.state_dict(), adrs)


if __name__ == "__main__":
    print("Main")