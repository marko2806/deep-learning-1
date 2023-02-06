from torch.utils.data import Dataset
from collections import defaultdict
import torchvision
from random import randint
import torch


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            indices_to_remove = []
            indices_to_preserve = list(range(len(self.images)))
            self.classes.remove(remove_class)
            for i in range(len(self.images)):
                if self.targets[i].item() == remove_class:
                    indices_to_remove += [i]
            for i in indices_to_remove:
                indices_to_preserve.remove(i)
            self.images = self.images[indices_to_preserve]
            self.targets = self.targets[indices_to_preserve]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]


    def _sample_negative(self, index):
        image_class = self.targets[index].item()
        possible_classes = self.classes.copy()
        possible_classes.remove(image_class)
        negative_index = randint(0, len(possible_classes) - 1)
        negative_class = possible_classes[negative_index]
        image_index = self.target2indices[negative_class][randint(0, len(self.target2indices[negative_class]) - 1)]
        return image_index

    def _sample_positive(self, index):
        image_class = self.targets[index].item()
        image_index = self.target2indices[image_class][randint(0, len(self.target2indices[image_class]) - 1)]
        return image_index

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = MNISTMetricDataset(remove_class=1)
    print(len(dataset.target2indices[1]))
