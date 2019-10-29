
def train(net, loader, loss_func, optimizer, local_ep, device="cpu"):

    net.train()
    # train and update
    epoch_loss = []
    epoch_acc = []
    avrg = lambda a: sum(a)/len(a)

    for iter in range(local_ep):
        batch_loss = []
        correct = 0

        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # net.zero_grad()

            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            y_pred = log_probs.data.max(1, keepdim=True)[1]

            correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        epoch_acc.append(correct.float()*100./len(loader.sampler))

    # Calculate loss average
    return net.state_dict(), avrg(epoch_loss), avrg(epoch_acc)


def test(net, loader, loss_func, device="cpu"):
    # testing
    net.eval()
    test_loss = []
    correct = 0
    avrg = lambda a: sum(a)/len(a)

    for idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)

        log_probs = net(data)
        loss = loss_func(log_probs, labels)
        test_loss.append(loss.item())

        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).sum()

    return avrg(test_loss), correct.float()*100./len(loader.sampler)

