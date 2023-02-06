import numpy as np
import torch
from sklearn.metrics import f1_score


def accuracy(gt, out):
    classes = (out >= 0).astype(np.int8)
    correct = np.sum(classes == gt)
    return correct / gt.shape[0] * 100


def f1(gt, out):
    classes = (out >= 0).astype(np.int8)
    return f1_score(gt, classes)


def confusion_matrix(gt, out, num_classes):
    classes = (out >= 0).astype(np.int8)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int16)
    for i in range(0, gt.shape[0]):
        matrix[gt[i], classes[i]] += 1
    return matrix


def evaluate(model, data, criterion=None, device="cpu", print_acc=True):
    model.eval()
    accuracies = []
    outputs = []
    gts = []
    with torch.no_grad():
        for i, (texts, labels, lenghts) in enumerate(data):
            out = model.forward(texts.to(device))
            outputs += out.cpu().detach().reshape(-1).tolist()
            gts += labels.cpu().tolist()
            accuracies.append((accuracy(labels.cpu().numpy(), out.cpu().detach().numpy())))
    gts = np.array(gts)
    outputs = np.array(outputs)
    if print_acc:
        print(f"Evaluation finished, accuracy: {accuracy(gts,outputs)}")
        #print(f"Confusion matrix: \n {confusion_matrix(gts, outputs, 2)}")
        #print(f"F1-Score {f1(gts, outputs)}")


def train(model, optimizer, criterion, data, device="cpu", use_clipping=False):
    model.train()
    for i, (texts, labels, lenghts) in enumerate(data):
        optimizer.zero_grad()
        logits = model.forward(texts.to(device))

        loss = criterion(logits.to(device), torch.reshape(labels.float(), (-1, 1)).to(device))
        loss.backward()
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        # losses.append(loss.cpu().detach().item())
        # accuracies.append(accuracy(labels.cpu().numpy(), logits.cpu().detach().numpy()))
    # print(f"Train finished, accuracy: {sum(accuracies) / len(accuracies) * 100}")


def train_(model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader, epochs=5,
           device="cpu", use_clipping=False):
    for i in range(epochs):
        losses = []
        accuracies = []
        for j, (texts, labels, lenghts) in enumerate(train_dataloader):
            optimizer.zero_grad()
            logits = model.forward(texts.to(device))
            loss = criterion(logits.to(device), torch.reshape(labels.float(), (-1, 1)).to(device))
            loss.backward()
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            losses.append(loss.cpu().detach().item())
            accuracies.append(accuracy(labels.cpu().numpy(), logits.cpu().detach().numpy()))
        print(f"Train finished, accuracy: {sum(accuracies) / len(accuracies) * 100}")
        accuracies = []
        for j, (texts, labels, lenghts) in enumerate(val_dataloader):
            with torch.no_grad():
                out = model.forward(texts.to(device))
                accuracies.append((accuracy(labels.cpu().numpy(), out.cpu().detach().numpy())))
        print(f"Val finished, accuracy: {sum(accuracies) / len(accuracies) * 100}")

    accuracies = []
    for i, (texts, labels, lenghts) in enumerate(test_dataloader):
        with torch.no_grad():
            out = model.forward(texts.to(device))
            accuracies.append((accuracy(labels.cpu().numpy(), out.cpu().detach().numpy())))
    print(f"Test finished, accuracy: {sum(accuracies) / len(accuracies) * 100}")
