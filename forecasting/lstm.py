import torch
import torch.nn as nn
import numpy as np


def unpack_nested_list(array_list):
    return np.hstack([item[0] for item in array_list])


def training(num_epochs, train_loader, optimizer, model, criterion):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            def closure():
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels.float())
                loss.backward()
                return loss
            optimizer.step(closure)
    return model


def testing(test_loader, model, criterion):
    y_hats = []
    ys = []
    for i, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            future = 1
            pred = model(inputs, future=future)
            loss = criterion(pred, labels)
            y_hat = pred.detach().numpy()
            y_hats.append(y_hat)
            y = labels.detach().numpy()
            ys.append(y)
            print(f"Test Loss {loss.item()}")
    predictions = unpack_nested_list(y_hats)
    labels = unpack_nested_list(ys)
    return labels, predictions


class FlatLSTM(nn.Module):
    def __init__(self, n_hidden, sequence_length):
        super(FlatLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(sequence_length, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, sequence_length)

    def forward(self, x, future=None):
        outputs = []
        ht1, ct1 = self.lstm1(x.float())
        ht2, ct2 = self.lstm2(ht1)
        output = self.linear(ht2)
        outputs.append(output)
        if future:
            predictions = []
            for i in range(future):
                ht1, ct1 = self.lstm1(output.float(), (ht1, ct1))
                ht2, ht2 = self.lstm2(ht1, (ht2, ct2))
                output = self.linear(ht2)
                predictions.append(output)
            return torch.cat(predictions, dim=1)

        return torch.cat(outputs, dim=1)

