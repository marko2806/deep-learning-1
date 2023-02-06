import torch
import numpy as np
from pathlib import Path
import  torch.nn.functional as F
import conv_utils
import math

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'task3'

epochs = 10
batch_size = 50
weight_decay = 1e-3
lr_policy = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.relu2 = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(512, 512)
        self.relu3 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(512, 10)
    def forward(self, x):
        h = x
        h = self.conv1.forward(h)
        h = self.maxpool1.forward(h)
        h = self.relu1.forward(h)

        h = self.conv2.forward(h)
        h = self.maxpool2.forward(h)
        h = self.relu2.forward(h)

        h = self.flatten.forward(h)

        h = self.fc1.forward(h)
        h = self.relu3.forward(h)
        
        h = self.fc2.forward(h)
        return h
    def loss(self, x, y_):
        mean = -torch.mean(y_ * torch.log(torch.softmax(x, 1)))
        return mean
    def one_item_loss(self, x, y_):
        loss = y_ * torch.log(torch.softmax(x, 0))
        return -torch.mean(loss)

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

def train(model, train_data, val_data, device=torch.device("cpu")):
    train_x, train_y = train_data
    train_x, train_y = torch.tensor(train_x).float().to(device), torch.tensor(train_y).float().to(device)
    num_examples = train_x.shape[0]
    num_of_batches = math.ceil(num_examples / batch_size)

    losses = []
    optimizer = torch.optim.SGD([
            {"params": model.conv1.parameters()}, 
            {"params": model.conv2.parameters()}, 
            {"params": model.fc1.parameters()}, 
            {"params": model.fc2.parameters(), "weight_decay": 0.0}
        ], lr=1e-1, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_policy.keys(), gamma=0.1)
    epoch_losses = []
    #train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]
        for j in range(num_of_batches):
            batch_x = train_x[j * batch_size: (j + 1) * batch_size] 
            batch_y = train_y[j * batch_size: (j + 1) * batch_size] 
            batch_y = F.one_hot(batch_y.long(), num_classes= 10)

            logits = model.forward(batch_x)
            #print(logits.size())
            #print(batch_y.size())
            loss = model.loss(logits, batch_y)
            loss.backward()
            
            # optimizacija
            optimizer.step()
            # reset gradijenta
            optimizer.zero_grad()

            epoch_losses.append(loss.cpu().detach().numpy())
        epoch_loss = np.mean(np.array(epoch_losses))
        losses.append(epoch_loss)
        plot_data['lr'].append(scheduler.get_last_lr()) 
        scheduler.step()
        # reset gradijenta
        train_loss, train_acc, train_p, train_r = conv_utils.evaluate(model, train_data, "Train", batch_size, device)
        val_loss, val_acc, val_p, val_r = conv_utils.evaluate(model, val_data, "Validation", batch_size, device)
        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        print(f"Epoch {epoch} finished")
    conv_utils.plot_training_progress(SAVE_DIR, plot_data)
