import pandas as pd
import copy


# 1. Dataset is separeted in iid
# input - frame
# return - workers[1,2,3,4]:[Dataframe]

def split_iid(frame, n_workers):
    label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
    amount = (label_count / n_workers).astype(int)
    print("Each worker get", amount, "of samples")
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
def split_by_attack(frame):
    label_count = frame.groupby("URL_Type_obf_Type")["Querylength"].count()
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
def split_identical(frame, n_workers):
    workers = dict.fromkeys(range(n_workers), pd.DataFrame())
    for worker_id in range(n_workers):
        workers[worker_id] = copy.copy(frame)
    return workers
