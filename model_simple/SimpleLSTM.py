import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class SimpleLSTM(nn.Module):
    def __init__(self, n_input, n_embed, n_hidden, n_output, n_layers, device):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(n_input, n_embed)
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, n_output)

        self.n_input = n_input
        self.n_embed = n_embed
        self.n_output = n_output
        self.n_layers = n_layers
        self.device = device

    def forward(self, x, x_len):
        embedded = self.embedding(x)
        emb_packed = pack_padded_sequence(embedded, x_len, batch_first=True)
        packed_out, _ = self.lstm(emb_packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        output = self.linear(lstm_out)
        return output
