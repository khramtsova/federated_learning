import numpy as np
import torch

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader, SubsetRandomSampler


# Imagenet normalization:
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])


def create_mask(trainset, classes):
    mask = [a in classes for a in trainset.targets]
    indices = [i for i, x in enumerate(mask) if x]
    return mask, indices #torch.tensor(mask)


# iterable function
def change_target_indx(it, subclasses):
    return subclasses.index(it)


def get_data(name: str, batch_size, num_workers, sub_sample=None):

    if name == "mnist":
        transform = Compose(
            [ToTensor(),
             #Normalize((MNIST.mean() / 255,), (MNIST.std() / 255,)),
             Resize((224, 224))])
        trainset = MNIST(download=False,
                         train=True,
                         root=".",
                         transform=transform)
        testset = MNIST(download=False,
                        train=False,
                        root=".",
                        transform=transform)
        #labels_train = MNIST(download=False, train=True, root=".").targets.float()
        #labels_test = MNIST(download=False, train=False, root=".").targets.float()

    if name == "cifar":
        transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(size=24),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = CIFAR10(download=False,
                           train=True,
                           root=".",
                           transform=transform)
        testset = CIFAR10(download=False,
                          train=False,
                          root=".",
                          transform=transform)
        #labels_train = CIFAR10(download=False, train=True, root=".").targets
        #labels_test = CIFAR10(download=False, train=False, root=".").targets
    if sub_sample is not None:
        train_mask, train_indices = create_mask(trainset, sub_sample)
        test_mask, test_indices = create_mask(testset, sub_sample)

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Replace indexes with range(len(classes))
        trainset.targets = np.array(trainset.targets)
        testset.targets = np.array(testset.targets)
        masked_label_train = trainset.targets[train_mask]
        masked_label_test = testset.targets[test_mask]
        trainset.targets[train_mask] = [sub_sample.index(it) for it in masked_label_train]
        testset.targets[test_mask] = [sub_sample.index(it) for it in masked_label_test]

    else:
        train_sampler = None
        test_sampler = None

    #print("Labels", np.unique(trainset.targets))
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=False,
                             sampler=train_sampler,
                             num_workers=num_workers)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=test_sampler,
                            num_workers=num_workers)

    return trainloader, testloader


if __name__ == "__main__":
    trainset = MNIST(download=False,
                     train=True,
                     root=".")
    ind = [a in [5,0] for a in trainset.targets]
    print(trainset)
    print(np.shape(trainset.data[ind]))

    transform = Compose(
        [ToTensor(),
         # Normalize((MNIST.mean() / 255,), (MNIST.std() / 255,)),
         Resize((224, 224))])
