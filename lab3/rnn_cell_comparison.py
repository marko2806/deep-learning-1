import random

from NLPDataset import NLPDataset
from embedding import build_embedding_matrix
import torch
import torch.nn as nn
from rnn import RNNModel
from torch.utils.data import DataLoader
from dataset import pad_collate_fn
from utils import train, evaluate
import itertools


if __name__ == "__main__":
    device = "cuda"
    rnn_types = ["rnn", "gru", "lstm"]
    bidirectionals = [True, False]
    hidden_sizes = [50, 150, 300]
    num_layers = [2, 5, 10]
    dropouts = [0, 0.25, 0.5]

    train_dataset = NLPDataset.from_file('dataset/sst_train_raw.csv')
    test_dataset = NLPDataset.from_file('dataset/sst_test_raw.csv')
    val_dataset = NLPDataset.from_file('dataset/sst_valid_raw.csv')

    text_vocab, label_vocab = NLPDataset.build_vocabulary(train_dataset)
    test_dataset.text_vocab = text_vocab
    test_dataset.label_vocab = label_vocab
    val_dataset.text_vocab = text_vocab
    val_dataset.label_vocab = label_vocab

    e_matrix = build_embedding_matrix(text_vocab)
    for i in range(15):

        rnn_type = rnn_types[i // 5]
        bidirectional = True if random.randint(0, 1) == 1 else False
        hidden_size = hidden_sizes[random.randint(0, 2)]
        num_layer = num_layers[random.randint(0, 2)]
        dropout = dropouts[random.randint(0, 2)]

        print(f"rnn_type = {rnn_type} "
              + f"bidirectional = {bidirectional} "
              + f"hidden_site = {hidden_size} "
              + f"num_layers = {num_layer} "
              + f"dropout = {dropout}")
        model = RNNModel(e_matrix, rnn_type=rnn_type, hidden_size=hidden_size,
                         dropout=dropout, num_layers=num_layer,
                         bidirectional=bidirectional).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.BCEWithLogitsLoss().to(device)

        train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=pad_collate_fn, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate_fn, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=pad_collate_fn, shuffle=True)

        for epoch in range(5):
            train(model, optimizer, criterion, train_dataloader, use_clipping=False, device=device)
            #print("Evaluating on val set")
            evaluate(model, val_dataloader, device=device, print_acc=False)
        #print("Evaluating on test set")
        evaluate(model, test_dataloader, device=device)