import torch
from instance import Instance
from vocab import Vocab


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.label_vocab = None
        self.text_vocab = None
        self.instances = []

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        instance: Instance = self.instances[item]
        if instance is not None:
            num_text = self.text_vocab.encode(instance.value)
            num_label = self.label_vocab.encode(instance.label)
            return num_text, num_label

    @staticmethod
    def from_file(file_path):
        dataset = NLPDataset()
        with open(file_path, "r") as file:
            for index, line in enumerate(file):
                line = line.strip().split(", ")
                line[0] = line[0].split(" ")
                dataset.instances.append(Instance(line[0], line[1]))
        return dataset

    @staticmethod
    def build_vocabulary(dataset, min_freq=1, max_size=-1):
        text_freq = {}
        label_freq = {}
        text_freq["<PAD>"] = float("inf")
        text_freq["<UNK>"] = float("inf")
        for instance in dataset.instances:
            for i in range(0, len(instance.value)):
                if instance.value[i] not in text_freq:
                    text_freq[instance.value[i]] = 1
                else:
                    text_freq[instance.value[i]] += 1
            if instance.label not in label_freq:

                label_freq[instance.label] = 1
            else:
                label_freq[instance.label] += 1
        dataset.text_vocab = Vocab(list(text_freq.items()), max_size, min_freq)
        dataset.label_vocab = Vocab(list(label_freq.items()), max_size, min_freq)
        return dataset.text_vocab, dataset.label_vocab


if __name__ == "__main__":
    train_dataset = NLPDataset.from_file("dataset/sst_train_raw.csv")
    text_vocab, label_vocab = NLPDataset.build_vocabulary(train_dataset)
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    # >> > Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
    # >> > Label: positive
    print(f"Numericalized text: {text_vocab.encode(instance_text)}")
    print(f"Numericalized label: {label_vocab.encode(instance_label)}")
    # >> > Numericalized
    # text: tensor([189, 2, 674, 7, 129, 348, 143])
    # >> > Numericalized
    # label: tensor(0)

    print(len(text_vocab.itos))
    # 14806

