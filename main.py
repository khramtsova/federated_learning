
import argparse

import numpy as np

import matplotlib.pyplot as plt
import torch

from get_data import get_data
from model import Net
import torch.optim as optim
import torch.nn as nn

batch_size = 200
num_workers = 2
classes = [1,2,3,4,5,6,7,8,9]


def main():

    # Initialize the visualization environment
    accuracy = []
    train_loss = []
    test_loss = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_data("cifar",
                                       batch_size=batch_size,
                                       sub_sample=classes,
                                       #sub_sample=torch.tensor(classes),
                                       num_workers=num_workers)
    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    # show images
    #imshow(torchvision.utils.make_grid(images))

    print("Number of classes: ", len(classes))
    net = Net(len(classes))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_loss = []
        net.train()

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_loss.append(loss.item())

            # print statistics
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d, %d] loss: %.3f' %
                      (epoch + 1, i + 1, epoch*(400) +i+1, running_loss / 200))
                running_loss = 0.0

        # Calculate loss average
        loss_avrg = sum(batch_loss)/len(batch_loss)
        train_loss.append(loss_avrg)
        #print("Train loss", loss_avrg)
        #print('Finished Training')

        correct = 0
        total = 0
        with torch.no_grad():
            batch_loss = []
            net.eval()
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                batch_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += batch_size
                correct += (predicted == labels).sum().item()
            #print("Accumulated test loss", loss_avrg)
            #print("Accuracy", correct, "out of", total)

        #print("TEST LOSS", loss_avrg)
        test_loss.append(sum(batch_loss)/len(batch_loss))
        accuracy.append(correct / total)
        #print('Accuracy of the network on test images: %d %%' % (
        #        100 * correct / total))

    # ====================== PLOT ==========================
    plt.plot(accuracy, label='Validation accuracy')
    plt.legend(frameon=False)
    plt.savefig("./logs/accuracy.png")
    plt.clf()

    plt.plot(train_loss, label='Train loss')
    plt.plot(test_loss, label='Test loss')
    plt.legend(frameon=False)
    plt.savefig("./logs/loss.png")

    # ====================== PLOT ==========================
    print("Saving model to ./logs")
    adrs = "./logs/model" + "".join(map(str, classes)) +".pth"
    torch.save(net.state_dict(), adrs)


if __name__ == '__main__':
    main()
