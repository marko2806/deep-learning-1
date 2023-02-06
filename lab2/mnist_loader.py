from pathlib import Path
import pt_model_3
import conv_utils
from torchvision.datasets import MNIST
import torch

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'task3'


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:   
    device = torch.device('cpu')

with torch.cuda.device("cuda:0"):

    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x_n, valid_x_n, test_x_n = train_x, valid_x, test_x
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    
    model = pt_model_3.Model().to(device)

    conv_utils.draw_conv_filters(0, 0, model.conv1.weight.cpu().detach().numpy(), SAVE_DIR)
    pt_model_3.train(model, (train_x, train_y), (valid_x, valid_y), device)
    conv_utils.draw_conv_filters(10, 0, model.conv1.weight.cpu().detach().numpy(), SAVE_DIR)
    conv_utils.evaluate(model, (test_x, test_y), "Test", 50, device)

    
    conv_utils.show_images_with_biggest_loss(model, (train_x_n, train_y), 0, 255)
