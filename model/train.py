import torch


def train(worker, net_in, loader, loss_func, local_ep, batch_size, optim="Adam",device="cpu"):

    net = net_in.copy()

    if optim == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        if optim == "SGD":
            # ToDo add momentum
            optimizer = optim.SGD(net.parameters(), lr=0.001)#, momentum=args.momentum)
        else:
            raise Exception("Unknown optimizer")

    # ===== Federated part =====
    # Send the model to the right location
    net.send(worker)
    net.train()

    epoch_loss = []
    epoch_acc = []

    avrg = lambda a: sum(a)/len(a)

    for iter in range(local_ep):

        batch_loss = []# torch.Tensor(len(loader)).send(worker)
        correct = 0

        for indx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

            loss.backward()
            optimizer.step()
            loss = loss.get()
            batch_loss.append(loss.item())

        epoch_loss.append(avrg(batch_loss))
        #print("Loss after epoch", iter, "is", avrg(batch_loss))

        # Get the accuracy back, specify that is float and transform from Tensor to value
        correct_value = correct.get().float().item()
        epoch_acc.append(correct_value*100./(len(loader)*batch_size))

    # Calculate loss average

    return net.get(), avrg(epoch_loss), avrg(epoch_acc)


def test(worker, net_in, loader, loss_func, batch_size, device="cpu"):
    net = net_in.copy()
    # testing
    net.send(worker)
    net.eval()
    test_loss = []
    correct = 0
    avrg = lambda a: sum(a)/len(a)
    with torch.no_grad():
        for idx, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)

            log_probs = net(data)
            loss = loss_func(log_probs, labels)

            loss = loss.get()
            test_loss.append(loss.item())

            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

    correct_value = correct.get().float().item()
    accuracy = correct_value*100./(len(loader)*batch_size)

    return avrg(test_loss), accuracy

