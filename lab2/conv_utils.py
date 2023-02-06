import math
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)

def draw_image(img, mean, std):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)
  plt.show()

def show_images_with_biggest_loss(model, data, mean, std):
    data_x, data_y = data
    losses = np.zeros((data_x.shape[0]))
    model = model.to(torch.device("cpu"))
    predictions = np.zeros((data_x.shape[0]))
    with torch.no_grad():
        logits = model.forward(torch.tensor(data_x).float())

        y_oh = F.one_hot(torch.tensor(data_y).long())
        for i in range(0, data_x.shape[0]):
            losses[i] = model.one_item_loss(logits[i, :], y_oh[i, :]).detach().numpy()
        predictions = np.argmax(logits, axis=1)
    idx = np.argsort(-losses)[0:20]
    for i in idx:
        draw_image(data_x[i], mean, std)
        print(f"Image {i} losses: {losses[i]}, prediction: {predictions[i]}, correct_class: {data_y[i]}")


def evaluate(model, data, data_type, batch_size, device=torch.device('cpu')):
    cm = np.zeros((10, 10), dtype="int32")
    num_batches =  math.ceil(data[0].shape[0] / batch_size)
    loss = 0.0
    precisions = np.zeros((10,))
    recalls = np.zeros((10,))

    with torch.no_grad():
        for i in range(num_batches):
            batch_x = torch.tensor(data[0][i * batch_size: (i + 1) * batch_size, :]).float().to(device)
            batch_y = torch.tensor(data[1][i * batch_size: (i + 1) * batch_size]).long().to(device)
            logits = model.forward(batch_x)

            pred = torch.argmax(logits, dim=1)
            batch_loss = model.loss(logits, F.one_hot(batch_y, num_classes=10))
            loss += batch_loss.cpu().detach().numpy()

            y_true = batch_y.cpu().detach().numpy()
            y_pred = pred.cpu().detach().numpy()

            cm += confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])
            precisions += precision_score(y_true, y_pred, average=None, labels=[i for i in range(10)], zero_division=0)
            recalls += recall_score(y_true, y_pred, average=None, labels=[i for i in range(10)], zero_division=0)

    precisions /= num_batches
    recalls /= num_batches
    loss /= num_batches

    print(f"{data_type} - Confusion matrix: \n {cm}")
    
    num_elements = np.sum(cm)
    tp = np.sum(np.diagonal(cm))

    accuracy = tp / num_elements

    print(f"{data_type} - Accuracy: {accuracy}")
    print(f"{data_type} - Precisions: {precisions}")
    print(f"{data_type} - Recalls: {recalls}")
    return loss, accuracy, precisions, recalls