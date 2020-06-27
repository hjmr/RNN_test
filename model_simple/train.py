import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from SimpleLSTM import SimpleLSTM


def parse_arg():
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='the training epochs.')
    parser.add_argument('-i', '--n_input', type=int, default=10,
                        help='the number of input neurons.')
    parser.add_argument('-m', '--n_embed', type=int, default=10,
                        help='the dimension of the embedding vector.')
    parser.add_argument('-n', '--n_hidden', type=int, default=10,
                        help='the number of hidden neurons in LSTM.')
    parser.add_argument('-l', '--n_layers', type=int, default=1,
                        help='the number of layers of LSTM.')
    return parser.parse_args()


def validation(model, loss_func, inputs, inp_len, targets):
    avg_loss = 0
    with torch.no_grad():
        y = model(inputs, inp_len)
        avg_loss += loss_func(y[-1], targets)
    return avg_loss / len(targets)


def output_log(epoch, loss):
    print("{}, {}".format(epoch, loss), flush=True)


def prepare_data(x, y, device):
    x_len = torch.tensor([len(v) for v in x], dtype=torch.long, device=device)
    x_list = [torch.tensor(v, dtype=torch.long, device=device) for v in x]
    x_list = pad_sequence(x_list, batch_first=True)
    x_len, perm_idx = x_len.sort(0, descending=True)
    x_list = x_list[perm_idx]
    y_list = torch.tensor(y, dtype=torch.float, device=device)
    y_list = y_list[perm_idx]
    return x_list, x_len, y_list


def main(epoch, n_input, n_embed, n_hidden, n_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_list = [[1, 2, 3, 4], [4, 5, 6], [7, 8, 9], [2, 3, 4, 5]]
    y_list = [[0], [1], [0], [1]]
    inputs, inp_len, targets = prepare_data(x_list, y_list, device)

    n_output = len(targets[0])

    model = SimpleLSTM(n_input, n_embed, n_hidden, n_output, n_layers, device).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    output_log('start', validation(model, loss_func, inputs, inp_len, targets))

    for epoch in range(args.epoch):
        model.zero_grad()
        y = model(inputs, inp_len)
        loss = loss_func(y[-1], targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            output_log(epoch+1, validation(model, loss_func, inputs, inp_len, targets))

    output_log('last', validation(model, loss_func, inputs, inp_len, targets))


if __name__ == '__main__':
    args = parse_arg()
    main(args.epoch, args.n_input, args.n_embed, args.n_hidden, args.n_layers)
