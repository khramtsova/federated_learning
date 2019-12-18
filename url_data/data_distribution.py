import pandas as pd
import copy


class Distribute:
    def __init__(self, num_agents, num_classes=5):
        self.n_workers = num_agents

    # return - workers[1,2,3,4]:[Dataframe]
    def perform_split(self, name, frame):

        workers = dict((i, pd.DataFrame()) for i in range(self.n_workers))
        distribution = dict((i, pd.Series({"Defacement": 0,
                                           "benign": 0,
                                           "malware": 0,
                                           "phishing": 0,
                                           "spam": 0}))
                            for i in range(self.n_workers))
        if name == "iid":
            return self._split_iid(frame, workers, distribution)
        else:
            if name == "by_attack":
                return self._split_by_attack(frame, workers, distribution)
            else:
                if name == "same":
                    return self._split_identical(frame, workers)
                else:
                    raise ("Unknown image_data distribution", name)

    # 1. Dataset is separeted in iid
    # input - frame
    # return - workers[1,2,3,4]:[Dataframe]
    def _split_iid(self, frame, workers, distribution):

        label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
        distribution = (label_count / self.n_workers).astype(int)

        #for worker_id in range(self.n_workers):
        #    distribution[worker_id] = amount

        # For each type of the attack
        for name, subframe in frame.groupby("URL_Type_obf_Type"):
            for worker_id in range(self.n_workers):
                # Random sample of data
                temp = subframe.sample(n=distribution[name])
                workers[worker_id] = workers[worker_id].append(temp)
                subframe = subframe.drop(temp.index)
        return workers, distribution

    # 1. Dataset is separeted in iid
    # input - frame
    # return - workers[1,2,3,4]:[Dataframe]
    def _split_iid_balanced(self, frame, workers, distribution):

        label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
        distribution = (label_count / self.n_workers).astype(int)

        #for worker_id in range(self.n_workers):
        #    distribution[worker_id] = amount

        # For each type of the attack
        for name, subframe in frame.groupby("URL_Type_obf_Type"):
            for worker_id in range(self.n_workers):
                # Random sample of data
                temp = subframe.sample(n=distribution[name])
                workers[worker_id] = workers[worker_id].append(temp)
                subframe = subframe.drop(temp.index)
        return workers, distribution

    # 2. Dataset is split by attack, benign is heavily undersampled
    # input - frame
    # return - workers[1,2,3,4]:[Dataframe]
    def _split_by_attack(self, frame, workers, distribution):

        label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
        amount = (label_count["benign"] / 4).astype(int)

        benign = frame[frame["URL_Type_obf_Type"] == "benign"]
        # Frame doesn't contain benign anymore
        frame = frame.drop(benign.index)

        # For each worker
        for worker_id, (name, subframe) in enumerate(frame.groupby("URL_Type_obf_Type")):
            # Assign all the attacks
            workers[worker_id] = workers[worker_id].append(subframe)

            # Distribute benign iid among workers
            temp = benign.sample(n=amount)
            benign = benign.drop(temp.index)
            workers[worker_id] = workers[worker_id].append(temp)

            distribution[worker_id]["benign"] = amount
            distribution[worker_id][name] = len(subframe)
        return workers, distribution

    # 3. Dataset is the same for everyone
    # input - frame
    # return - workers[1,2,3,4]:[frame, frame, frame, frame]
    def _split_identical(self, frame, workers):
        for worker_id in range(self.n_workers):
             workers[worker_id] = copy.copy(frame)
        distribution = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
        return workers, distribution



# 1. Dataset is separeted in iid
# input - frame
# return - workers[1,2,3,4]:[Dataframe]
def split_iid(frame, n_workers, logs):
    label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
    amount = (label_count / n_workers).astype(int)
    print("Each worker get", amount)
    # workers = dict.fromkeys([0,1,2,3], pd.DataFrame())
    workers = dict.fromkeys(range(n_workers), pd.DataFrame())

    # For each type of the attack
    for name, subframe in frame.groupby("URL_Type_obf_Type"):

        for worker_id in range(n_workers):
            # Random sample of data
            temp = subframe.sample(n=amount[name])
            workers[worker_id] = workers[worker_id].append(temp)
            subframe = subframe.drop(temp.index)

    return workers


# 2. Dataset is split by attack, benign is heavely undersampled
# input - frame
# return - workers[1,2,3,4]:[Dataframe]
def split_by_attack(frame, logs):
    label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
    print("HERE",label_count)
    amount = (label_count["benign"] / 4).astype(int)

    benign = frame[frame["URL_Type_obf_Type"] == "benign"]
    # Frame doesn't contain benign anymore
    frame = frame.drop(benign.index)

    # Create empty array of workers
    workers = dict.fromkeys([0, 1, 2, 3], pd.DataFrame())

    # For each worker

    for worker_id, (name, subframe) in enumerate(frame.groupby("URL_Type_obf_Type")):
        # Assign all the attacks
        workers[worker_id] = workers[worker_id].append(subframe)

        # Distribute benign iid among workers
        temp = benign.sample(n=amount)
        benign = benign.drop(temp.index)
        workers[worker_id] = workers[worker_id].append(temp)

    return workers


# 3. Dataset is the same for everyone
# input - frame
# return - workers[1,2,3,4]:[frame, frame, frame, frame]
def split_identical(frame, n_workers, logs):
    workers = dict.fromkeys(range(n_workers), pd.DataFrame())
    for worker_id in range(n_workers):
        workers[worker_id] = copy.copy(frame)
    return workers
