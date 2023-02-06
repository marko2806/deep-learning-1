import torch


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        self.itos = {}
        self.stoi = {}
        frequencies = sorted(frequencies, key=lambda x: x[1], reverse=True)
        ctr = 0
        for token, frequency in frequencies:
            if frequency >= min_freq or min_freq == 0:
                self.itos[ctr] = token
                self.stoi[token] = ctr
                ctr += 1
            if max_size != -1 and len(self.itos) >= max_size:
                break

    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi[tokens])
        else:
            # vraca se indeks rjeci ako postoji, a inace <UNK>
            return torch.tensor([self.stoi[word] if word in self.stoi else 1 for word in tokens])
