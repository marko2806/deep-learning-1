import torch
import numpy as np

from NLPDataset import NLPDataset


def build_embedding_matrix(vocabulary, use_pretrained=True):
    if use_pretrained:
        matrix = torch.zeros((len(vocabulary.itos), 300))
        with open("dataset/sst_glove_6b_300d.txt", "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].split(" ")
                lines[i] = lines[i][0], lines[i][1:]

        i = 0
        for key in vocabulary.stoi:
            if key != "<PAD>":
                found = False
                for token, vector in lines:
                    if token == key:
                        found = True
                        for j in range(len(vector)):
                            vector[j] = float(vector[j])
                        matrix[i][:] = torch.tensor(vector).detach().clone()
                        break
                if not found:
                    matrix[i][:] = torch.randn((1, 300))
            i += 1
        return matrix
    else:
        matrix = torch.randn((len(vocabulary.itos), 300))
        #matrix[0][:] = torch.zeros((1, 300)).detach().clone()
        return matrix

if __name__ == "__main__":
    train_dataset = NLPDataset.from_file("dataset/sst_train_raw.csv")
    NLPDataset.build_vocabulary(train_dataset)
    instance_text, instance_label = train_dataset.instances[3]
    # Referenciramo atribut klase pa se ne zove nadjačana metoda
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    # >> > Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
    # >> > Label: positive
    numericalized_text, numericalized_label = train_dataset[3]
    # Koristimo nadjačanu metodu indeksiranja
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")
    # >> > Numericalized text: tensor([189, 2, 674, 7, 129, 348, 143])
    # >> > Numericalized label: tensor(0)
