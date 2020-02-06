import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# input data
x_list = [[1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
          [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
          [1, 1, 2, 1, 1, 2, 1, 1, 2]]
x_list = [torch.tensor(x, dtype=torch.long) for x in x_list]

# Word Embeddings (Word2Vec)
n_vocab = 10
emb_dim = 10
input_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=0)

# Parameters
in_size = emb_dim
hidden_size = 10
max_length = 12

# Encoder
n_enc_hidden = hidden_size
n_enc_layers = 1
n_latent = 2

enc_lstm = nn.LSTM(input_size=in_size,
                   hidden_size=n_enc_hidden,
                   num_layers=n_enc_layers,
                   batch_first=True)
enc_mu = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)
enc_ln_var = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)

# Decoder
n_dec_input = 1  # dummy
n_dec_layers = 1
n_dec_hidden = hidden_size

gen_z_h = nn.Linear(n_latent, n_dec_layers * n_dec_hidden)
dec_lstm = nn.LSTM(input_size=n_dec_input,
                   hidden_size=n_dec_hidden,
                   num_layers=n_dec_layers,
                   batch_first=True)
dec_out = nn.Linear(n_dec_hidden, in_size)

# prepare data
x_len = torch.LongTensor([len(x) for x in x_list])
x_data = pad_sequence(x_list, batch_first=True)
x_len, perm_idx = x_len.sort(0, descending=True)
x_data = x_data[perm_idx]

embedded = input_embed(x_data)
n_batch = embedded.size(0)
packed = pack_padded_sequence(embedded, x_len, batch_first=True)

_, (h_enc, _) = enc_lstm(packed)

mu = enc_mu(h_enc.view(n_batch, n_enc_layers * n_enc_hidden))
ln_var = enc_ln_var(h_enc.view(n_batch, n_enc_layers * n_enc_hidden))
z = torch.normal(mu, ln_var)

h_dec = gen_z_h(z)

y_inp = torch.ones(n_batch, max_length, n_dec_input)
dec_c0 = torch.zeros(n_dec_layers, n_batch, n_dec_hidden, requires_grad=True)
y_dec, _ = dec_lstm(y_inp, (h_dec.view(n_dec_layers, n_batch, n_dec_hidden), dec_c0))
y_out = dec_out(y_dec)

print(y_out.shape)
