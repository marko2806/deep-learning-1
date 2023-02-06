import torch.nn as nn
import torch
from dataset import NLPDataset, pad_collate_fn
from torch.utils.data import DataLoader
from embedding import build_embedding_matrix
from utils import train, evaluate


class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, num_layers=2, rnn_type=None, bidirectional=False, dropout=0.0, freeze=True):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=0)

        self.rnn_type = rnn_type

        if rnn_type == "lstm":
            self.rnn1 = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == "gru":
            self.rnn1 = nn.GRU(input_size=300, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout)
        else:
            self.rnn1 = nn.RNN(input_size=300, hidden_size=hidden_size, num_layers=num_layers,
                               bidirectional=bidirectional, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 1)

    def forward(self, inp):
        x = self.embedding(inp)
        x = torch.transpose(x, 0, 1)
        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn1.forward(x)
            x = h[self.num_layers - 1]
        else:
            x, h = self.rnn1.forward(x)
            x = h[self.num_layers - 1]
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        return x


if __name__ == "__main__":
    device = "cuda"

    train_dataset = NLPDataset.from_file('dataset/sst_train_raw.csv')
    test_dataset = NLPDataset.from_file('dataset/sst_test_raw.csv')
    val_dataset = NLPDataset.from_file('dataset/sst_valid_raw.csv')

    text_vocab, label_vocab = NLPDataset.build_vocabulary(train_dataset, min_freq=1)
    test_dataset.text_vocab = text_vocab
    test_dataset.label_vocab = label_vocab
    val_dataset.text_vocab = text_vocab
    val_dataset.label_vocab = label_vocab

    e_matrix = build_embedding_matrix(text_vocab, use_pretrained=True)
    model = RNNModel(e_matrix, rnn_type="gru", hidden_size=150, freeze=True, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss().to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=pad_collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=pad_collate_fn, shuffle=True)

    for epoch in range(10):
        train(model, optimizer, criterion, train_dataloader, use_clipping=False, device=device)
        evaluate(model, val_dataloader, device=device)

    evaluate(model, test_dataloader, device=device)


