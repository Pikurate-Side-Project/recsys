import torch
from torch import nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):

    def __init__(self, n_users, n_urls, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.url_emb = nn.Embedding(n_urls, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*2, hidden, dropouts)))
        self.fc = nn.Linear(hidden[-1], 1)

    def generate_layers(self, n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)

        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield nn.Linear(n_factors, hidden[idx])
            else:
                yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropouts[idx])

            idx += 1

    def forward(self, users, urls):
        concat_features = torch.cat([self.user_emb(users), self.url_emb(urls)], dim=1)
        x = F.relu(self.hidden_layers(concat_features))

        out = torch.squeeze(torch.sigmoid(self.fc(x)))

        return out

    def predict(self, users, urls):
        out = self.forward(users, urls)
        return out