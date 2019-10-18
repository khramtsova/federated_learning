import numpy as np

from random import shuffle

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader, SubsetRandomSampler


# Imagenet normalization:
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

def get_dataloaders(tr_data_dstr, test_data_distr, dataset_name, train_batch_size, test_batch_size, num_workers, sub_sample=None):

    trainset, testset = _get_data(dataset_name)
    print("Total number in trainset", len(trainset))

    n_classes = max(len(set(trainset.targets)), len(set(testset.targets)))
    _check_users_validity(tr_data_dstr, n_classes)

    train_samplers = _create_samplers(trainset, tr_data_dstr)
    test_samplers = _create_samplers(testset, test_data_distr)

    print("The number of train samples per agent ",
          [len(s) for i, s in enumerate(train_samplers)])
    print("The number of test samples per agent ",
          [len(s) for i, s in enumerate(test_samplers)])

    trainloaders, testloaders = [], []

    for sampler in train_samplers:
        trainloaders.append(DataLoader(trainset,
                                       batch_size=train_batch_size,
                                       shuffle=False,
                                       sampler=sampler,
                                       num_workers=num_workers))

    for sampler in test_samplers:
        testloaders.append(DataLoader(testset,
                                      batch_size=test_batch_size,
                                      shuffle=False,
                                      sampler=sampler,
                                      num_workers=num_workers))

    return trainloaders, testloaders


def _create_samplers(data, users):

    classes = list(set(data.targets))
    user_inds = [[] for i in range(len(users))]
    user_samplers = []

    for i, prob in enumerate(np.transpose(users)):

        mask = [a == classes[i] for a in data.targets]
        index = [i for i, x in enumerate(mask) if x]

        index_len = len(index)
        # replace probability with # of examples
        prob = [int(a*index_len) for a in prob]
        shuffle(index)

        prev = 0
        for k, number in enumerate(prob):
            user_inds[k].extend(index[prev:prev+number])
            prev = prev+number

    for indeces in user_inds:
        user_samplers.append(SubsetRandomSampler(indeces))

    return user_samplers


def _get_data(name: str):

    if name == "mnist":
        transform = Compose(
            [#Resize((224, 224)),
             ToTensor()
             #Normalize((MNIST.mean() / 255,), (MNIST.std() / 255,)),
             ])
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

    return trainset,testset


"""     # Replace indexes with range(len(classes))
        trainset.targets = np.array(trainset.targets)
        testset.targets = np.array(testset.targets)
        masked_label_train = trainset.targets[train_mask]
        masked_label_test = testset.targets[test_mask]
        trainset.targets[train_mask] = [sub_sample.index(it) for it in masked_label_train]
        testset.targets[test_mask] = [sub_sample.index(it) for it in masked_label_test]
"""


def _check_users_validity(users, n_classes):
    assert np.shape(users)[-1] <= n_classes, \
        "There are more parameters in user than classes in dataset"

    assert np.all(np.count_nonzero(users, axis=1) > 1) , \
        "There must be at least two classes per user"

    assert np.all(np.sum(users, axis=0) <= 1),\
        "The probablilities must sum up to 1"
    return


if __name__ == "__main__":
    get_dataloaders([[0,0.1,0.2,0.3,1,1,1,1,1,1],
                     [1,0.9,0.8,0.7,0,0,0,0,0,0]
                     ],
                    "cifar",
                    200,
                     2)
