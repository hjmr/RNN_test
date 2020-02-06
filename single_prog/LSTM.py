import torch
import torch.nn as nn

# input data
x_list = [[0, 1, 2, 3], [4, 5, 6, 7], [7, 8, 9, 5]]

# teaching
y_list = [[0, 1], [1, 2], [2, 3]]

# Word Embeddings (Word2Vec)
n_vocab = 10
emb_dim = 10
word_embed = nn.Embedding(n_vocab, emb_dim)

# LSTM
use_dropout = 0.25
n_lstm_in = emb_dim
n_lstm_layers = 2
n_lstm_out = emb_dim
calc_lstm = nn.LSTM(n_lstm_in, n_lstm_out, n_lstm_layers, batch_first=True)

# Output
n_output = 2
calc_output = nn.Linear(n_lstm_out, n_output)

# convert to tensor
x_data = torch.tensor(x_list, dtype=torch.long)
y_data = torch.tensor(y_list, dtype=torch.float)

embeds = word_embed(x_data)

print("embed shape")
print(embeds.size())
for e in embeds:
    print(e.size)

n_batch = embeds.size(0)
lstm_out, (lstm_hid, lstm_stat) = calc_lstm(embeds.view(n_batch, len(x_list[0]), -1))

print("lstm_hid / lstm_stat shape")
print(lstm_hid.size(), lstm_stat.size())

print("lstm_out shape")
print(lstm_out.size())

out = calc_output(lstm_out.view(n_batch, len(x_list[0]), -1))

print("out shape")
print(out.size())

loss_function = nn.MSELoss()
loss = []
for y, y_hat in zip(out, y_list):
    loss.append(loss_function(y[-1], torch.tensor(y_hat, dtype=torch.float)))

print("loss")
print(loss)
