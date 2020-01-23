
import pandas as pd
import torch
import syft as sy

from url_data import data_distribution


def get_dataloaders(file, logs,
                    tr_data_dstr,
                    test_data_distr,
                    num_workers=4,
                    train_batch_size=16,
                    test_batch_size=16):

    dataset = pd.read_csv(file, low_memory=False, squeeze=True)

    # Create Virtual Workers
    hook = sy.TorchHook(torch)
    workers = []
    for idx in range(num_workers):
        workers.append(sy.VirtualWorker(hook, id="worker" + str(idx)))

    # Set aside the test dataset, which will be the same for all the workers
    train_data, test_data = _train_validation_split(dataset, 10)

    # If by_attack - ignore the number of workers
    if tr_data_dstr == "by_attack" or test_data_distr == "by_attack":
        num_workers = 4

    distr = data_distribution.Distribute(num_workers)

    train_data_subsets, train_distribution = distr.perform_split(tr_data_dstr, train_data)
    test_data_subsets, test_distribution = distr.perform_split(test_data_distr, test_data)

    logs.plot_distribution(train_distribution, "train_distribution")
    logs.plot_distribution(test_distribution, "test_distribution")

    fed_dataset_train = _distribute_among_workers(train_data_subsets, workers)
    fed_dataset_test = _distribute_among_workers(test_data_subsets, workers)

    fed_loader_train = sy.FederatedDataLoader(fed_dataset_train, batch_size=train_batch_size, shuffle=True)
    fed_loader_test = sy.FederatedDataLoader(fed_dataset_test, batch_size=test_batch_size, shuffle=True)

    return fed_loader_train, fed_loader_test, workers


#
def _distribute_among_workers(dataset, workers):
    datasets = []

    for i, data in dataset.items():
        x_train, y_train = _data_target_split(data)

        data = x_train.send(workers[i])
        targets = y_train.send(workers[i])
        datasets.append(sy.BaseDataset(data, targets))

    return sy.FederatedDataset(datasets)


# Take equal amount of all the features apart
# All the workers have a same validation subset
def _train_validation_split(frame, percent):
    label_count = frame.groupby("URL_Type_obf_Type")["Len_Query"].count()
    # amount = (label_count/4).astype(int)
    amount = (label_count * percent / 100).astype(int)
    for key in amount.keys():
        amount[key] = min(amount)
    amount["benign"] = 1

    # workers = dict.fromkeys([0,1,2,3], pd.DataFrame())
    # For each type of the attack
    test_data = pd.DataFrame()
    for name, subframe in frame.groupby("URL_Type_obf_Type"):
        test_data = test_data.append(subframe.sample(n=amount[name]))
        #test_data = test_data.append(subframe.sample(n=amount[name]))

    #test_data = test_data.drop(test_data[test_data["URL_Type_obf_Type"] == "benign"].index)
    frame = frame.drop(test_data.index)
    # for worker_id in range(4):
    # Random sample of data

    # workers[worker_id] = workers[worker_id].append(temp)
    # subframe = subframe.drop(temp.index)
    return frame, test_data


# Separate target into a separate column and make it binary. Return Tensors
def _data_target_split(dataset):

    target = dataset.pop("URL_Type_obf_Type")
    how_replace = dict(zip(["defacement", "benign", "malware", "phishing", "spam"],
                           [1, 0, 1, 1, 1]))
    target = target.map(how_replace)
    x = torch.from_numpy(dataset.values).float()
    y = torch.from_numpy(target.values)  # .long()

    return x, y