import torch
import torch.nn as nn
import numpy as np


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()

        self.add_module("bn", nn.BatchNorm2d(num_maps_in))
        self.add_module("relu", nn.ReLU())
        self.add_module("conv", nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32, power=2, margin=1.0):
        super().__init__()
        self.emb_size = emb_size
        self.p = power
        self.margin = margin

        self.bnr_relu_conv1 = _BNReluConv(input_channels, emb_size)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.bnr_relu_conv2 = _BNReluConv(emb_size, emb_size)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.bnr_relu_conv3 = _BNReluConv(emb_size, emb_size)

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()



    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.bnr_relu_conv1.forward(img)
        x = self.pooling1.forward(x)
        x = self.bnr_relu_conv2.forward(x)
        x = self.pooling2.forward(x)
        x = self.bnr_relu_conv3.forward(x)
        x = self.avg_pool.forward(x)
        x = self.flatten.forward(x)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        #triplet_loss = nn.TripletMarginLoss()
        #loss = triplet_loss(a_x, p_x, n_x)
        d1 = torch.linalg.norm(a_x - p_x, dim=1)
        d2 = torch.linalg.norm(a_x - n_x, dim=1)
        loss = torch.mean(torch.maximum(d1 - d2 + torch.tensor(self.margin), torch.tensor(0)))
        return loss


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.flatten1 = nn.Flatten()

    def get_features(self, img: torch.Tensor):
        feats = self.flatten1.forward(img)
        return feats


if __name__ == "__main__":
    import dataset
    from torch.utils.data import DataLoader

    ds_train = dataset.MNISTMetricDataset()
    train_loader = DataLoader(
        ds_train,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    for i, data in enumerate(train_loader):
        anchor, positive, negative, _ = data
        print(anchor.shape)
        model = SimpleMetricEmbedding(1).to("cuda")
        features = model.get_features(anchor.to("cuda"))
        print(features.shape)


