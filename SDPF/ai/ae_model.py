import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 4096
EMBEDDING_DIM = 64


class LstmEncoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(LstmEncoder, self).__init__()
        self.input_size, self.latent_dim = input_size, latent_dim

        self.rnn1 = nn.LSTM(input_size=self.input_size, hidden_size=2 * self.latent_dim, batch_first=True, dropout=0.2)
        self.rnn2 = nn.LSTM(input_size=2 * self.latent_dim, hidden_size=self.latent_dim, batch_first=True)
        self.executed_once = False

    def forward(self, x):
        x, (h, _) = self.rnn1(x)
        x, (h, c) = self.rnn2(x)
        if not self.executed_once:
            self.executed_once = True
            print(x.shape)
        return x


class LstmDecoder(nn.Module):
    def __init__(self, input_size, output_dim):
        super(LstmDecoder, self).__init__()
        self.input_size, self.output_dim = input_size, output_dim
        self.rnn1 = nn.LSTM(input_size=self.input_size, hidden_size=2 * self.input_size, batch_first=True, dropout=0.2)
        self.rnn2 = nn.LSTM(input_size=2 * self.input_size, hidden_size=4 * self.input_size, batch_first=True)
        # self.rnn2 = nn.LSTM(input_size=4 * self.input_size, hidden_size=2 * self.input_size,
        # batch_first=True)
        self.output_layer = nn.Linear(4 * self.input_size, output_dim)

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (h, c) = self.rnn2(x)
        x = self.output_layer(x)
        return x


class LstmAutoencoder(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, embedding_dim=EMBEDDING_DIM, device=DEVICE):
        super(LstmAutoencoder, self).__init__()
        self.encoder = LstmEncoder(input_size, embedding_dim).to(device)
        self.decoder = LstmDecoder(embedding_dim, input_size).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #         print(x)
        return x
