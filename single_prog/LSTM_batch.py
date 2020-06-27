import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = {}".format(device))
vocab_size = 2  # 語彙数 0 or 1
embed_size = 5  # embeddingのサイズ
hidden_size = 5  # lstmの隠れ層のサイズ
embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)  # embedding
lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)  # LSTM(2,7,2) →(2,7,5)
embedding.to(device)
lstm.to(device)

# 入力データ準備
inputs = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]  # (2,7)のデータ ※0はpaddingを想定
lengths = [7, 5]  # (7,5)
inputs = torch.tensor(inputs, dtype=torch.long, device=device)  # tensorへ
lengths = torch.tensor(lengths, dtype=torch.long, device=device)  # tensorへ

embedded = embedding(inputs)  # (2,7,3) embedding
packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)  # embeddingしたデータと元の長さのlistを入れる
output, hidden = lstm(packed)  # 通常通りlstmに入れる
output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # lstmからの最後の隠れ状態を入れる

print("*** hidden")
print(hidden)

print("*** output")
print(output)

print("*** output_length")
print(output_lengths)
