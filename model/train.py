
def train(net, loader, loss_func, optimizer, local_ep, device="cpu"):

    net.train()
    # train and update
    epoch_loss = []
    for iter in range(local_ep):
        batch_loss = []

        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # net.zero_grad()

            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    # Calculate loss average
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

