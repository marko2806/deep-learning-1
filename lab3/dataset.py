from NLPDataset import NLPDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocab import Vocab
import torch
import embedding


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    # Process the text instances
    texts = pad_sequence([text for text in texts], batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels)
    return texts, labels, lengths


if __name__ == "__main__":

    batch_size = 2  # Only for demonstrative purposes
    shuffle = False  # Only for demonstrative purposes
    train_dataset = NLPDataset.from_file('dataset/sst_train_raw.csv')
    NLPDataset.build_vocabulary(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

    embedding.build_embedding_matrix(train_dataset.text_vocab)

    # >>> Texts: tensor([[   2,  554,    7, 2872,    6,   22,    2, 2873, 1236,    8,   96, 4800,
    #                       4,   10,   72,    8,  242,    6,   75,    3, 3576,   56, 3577,   34,
    #                    2022, 2874, 7123, 3578, 7124,   42,  779, 7125,    0,    0],
    #                   [   2, 2875, 2023, 4801,    5,    2, 3579,    5,    2, 2876, 4802,    7,
    #                      40,  829,   10,    3, 4803,    5,  627,   62,   27, 2877, 2024, 4804,
    #                     962,  715,    8, 7126,  555,    5, 7127, 4805,    8, 7128]])
    # >>> Labels: tensor([0, 0])
    # >>> Lengths: tensor([32, 34])
