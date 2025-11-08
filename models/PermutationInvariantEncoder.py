from torch import nn


class PermutationInvariantEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):  # (batch_size, N, D)
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.layer_2(x)

        x = x.mean(dim=1)  # (batch_size, D)

        return x
