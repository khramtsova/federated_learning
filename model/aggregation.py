import copy
import torch


def FedAvg(w, n):
    assert len(w) == len(n)
    N = sum(n)
    n = [x/N for x in n]
    print(n)
    w_avg = copy.deepcopy(w[0])
    for j in w_avg.keys():
        w_avg[j] *= n[0]
        for i in range(1, len(n)):
            w_avg[j] += w[i][j] * n[i]
    return w_avg



if __name__ == "__main__":
    model = torch.load("../logs/model01.pth")
    model2 = torch.load("../logs/model12.pth")
    models = [model, model2]
    # Transpose models

    b = FedAvg(models, [15,74])

    """FedAvg(models)
    print(model.keys())
    print(model['conv1.weight'].shape)
    print(model['conv2.weight'].shape)
    print(model['fc1.weight'].shape)
    print(model['fc3.weight'].shape)
    print(len(model))

    #for i in range(1,len())
    print("Hello")
    """