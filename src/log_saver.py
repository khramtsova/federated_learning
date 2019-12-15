
import os
import csv
import random
import string

import numpy as np
import matplotlib.pyplot as plt

import torch

from src.utils import get_project_root


class LogSaver:

    def __init__(self, args):

        self.args = args

        # ToDo change the name of the folder, random for now
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

        # Create a new folder for every experiment
        #self.folder = str(get_project_root()) + args.log_folder + random_str + "/")
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

        acc_train = np.reshape(acc_train, [self.args.rounds, self.args.num_workers])
        loss_train = np.reshape(loss_train, [self.args.rounds, self.args.num_workers])
        acc_test = np.reshape(acc_test, [self.args.rounds, self.args.num_workers])
        loss_test = np.reshape(loss_test, [self.args.rounds, self.args.num_workers])

        # ====================== PLOT ==========================

        for i, l in enumerate(loss_train.T):
            plt.plot(l, label='Agent ' + str(i))
        plt.legend(frameon=False)
        plt.title("Train loss")
        plt.savefig(self.folder + "loss_train.png")
        plt.clf()

        for i, l in enumerate(loss_test.T):
            plt.plot(l, label='Agent ' + str(i))
        plt.legend(frameon=False)
        plt.title("Test loss")
        plt.savefig(self.folder + "loss_test.png")
        plt.clf()

        for i, l in enumerate(acc_train.T):
            plt.plot(l, label='Agent ' + str(i))
        plt.legend(frameon=False)
        plt.title("Train accuracy")
        plt.savefig(self.folder+"acc_train.png")
        plt.clf()

        for i, l in enumerate(acc_test.T):
            plt.plot(l, label='Agent ' + str(i))
        plt.legend(frameon=False)
        plt.title("Test accuracy")
        plt.savefig(self.folder+ "acc_test.png")
        plt.clf()
        # ====================== PLOT ==========================

    def save_model(self, net):
        print("Saving model weights ")
        adrs = self.folder + "model" + ".pth"
        torch.save(net.state_dict(), adrs)


if __name__ == "__main__":
    print("Main")