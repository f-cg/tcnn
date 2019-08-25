import time

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_test(net, loader, classes, optimizer=None, criterion=None,
               class_tags=None, train_flag=True, cols=5, metric_saved=None,
               mode='test'):
    """
    classes: number of classes
    cols: number of columns for print
    return: total_acc
    """
    t1 = time.time()
    class_correct = torch.zeros(classes)
    class_total = [0]*classes
    prev = torch.is_grad_enabled()
    torch._C.set_grad_enabled(train_flag)
    net.training = train_flag
    running_loss = 0
    for data in loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        if train_flag:
            optimizer.zero_grad()
        outputs = net(images)
        if train_flag:
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    samples_tot = sum(class_total)
    if train_flag:
        mode = 'train'
        print('loss: %.3f' % (running_loss / samples_tot))
    if class_tags is None:
        class_tags = range(classes)
    class_correct = class_correct.numpy()
    class_accs = 100*class_correct/class_total
    if cols > 0:
        for i in range(classes):
            if class_total[i] != 0:
                tip = '{} acc {}: {:.2f}%'.format(
                    mode, class_tags[i], class_accs[i])
                print(tip, end='\t')
            if i % cols == cols-1:
                print()
    total_acc = 100 * class_correct.sum() / samples_tot
    if cols > 0:
        print('{} acc tot: {:.2f}%'.format(mode, total_acc), end='\t')
        print('Elapsed time:{:.2f}'.format(time.time()-t1))
    else:
        print(total_acc)
    if metric_saved is not None:
        metric_saved[mode]['class_accs'].append(class_accs)
        metric_saved[mode]['total_acc'].append(total_acc)
    torch._C.set_grad_enabled(prev)
    return total_acc
