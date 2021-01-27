import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import f1_score

class TwoLayersModel(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        '''

        Initialiser le modèle nn
        :param in_dim:
        :param out_dim:
        '''
        super(TwoLayersModel, self).__init__()

        self.ln1 = nn.Linear(in_dim, h_dim, bias=True)
        self.ln2 = nn.Linear(h_dim, out_dim, bias=True)
        self.ln2_drop = nn.Dropout(p=0.5)
        self.out = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        # Suffered from Dying RELU....
        X = F.relu(self.ln1(X))
        X = F.relu(self.ln2(X))
        X = F.relu(self.ln2_drop(X))
        X = self.out((X))
        return X


# Source : Notebook du prof
def collate(batch):
    # Ici je gère les batch pour la gestion des tensors dans les données
    # vous n'avez rien à faire
    transposed_data = list(zip(*batch))
    return torch.stack(transposed_data[0], 0), torch.stack(transposed_data[1], 0)

def train(train_X, train_Y, valid_X, valid_Y, epochs=100, batch_size=64, lr= 1e-1):

    model = TwoLayersModel(train_X.shape[1], int(train_X.shape[1] // 2), 4)

    #opti = torch.optim.SGD(model.parameters(), lr=lr)
    opti = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()

    # Transformer ndarray en torch.tensor
    t_train_X = torch.tensor(train_X, dtype=torch.float)
    t_train_Y = torch.tensor(train_Y, dtype=torch.long)

    t_valid_X = torch.tensor(valid_X, dtype=torch.float)
    t_valid_Y = torch.tensor(valid_Y, dtype=torch.long)

    train_set = data.TensorDataset(t_train_X, t_train_Y)
    valid_set = data.TensorDataset(t_valid_X, t_valid_Y)

    # Trier les données de façon aléatoire
    train_sampler = data.BatchSampler(data.RandomSampler(range(train_X.shape[0])), batch_size, False)
    valid_sampler = data.BatchSampler(data.SequentialSampler(range(valid_X.shape[0])), valid_X.shape[0], False)
    train_loader = data.DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate)
    valid_loader = data.DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=collate)

    losses = []
    f1_valid = []

    for e in range(epochs):
        model.train()

        for batch_ndx, (trn_x, trn_y) in enumerate(train_loader):
            y_pred = model(trn_x)
            loss = criterion(y_pred, trn_y)
            losses.append(loss)

            opti.zero_grad()
            loss.backward()
            opti.step()

        model.eval()
        for batch_ndx, (vld_x, vld_y) in enumerate(valid_loader):
            valid_y_pred = model(vld_x)
            valid_y_pred = np.argmax(valid_y_pred.detach().numpy(), axis=1)
            score = f1_score(vld_y, valid_y_pred, average="micro")
            f1_valid.append(score)

        print("Epoch {0}/{1} : Loss = {2:.2f}\tF1_score = {3:.2f}".format(e, epochs, losses[-1], f1_valid[-1]))

    return model, losses, f1_valid
