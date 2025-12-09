import torch
from torch import nn


def retanh(x):
    """Rectified tanh activation function."""
    return torch.maximum(torch.tanh(x), torch.tensor(0))


class CTRNN(nn.Module):
    """Continuous-Time Recurrent Neural Network (CTRNN) model."""

    def __init__(self, n_input: int, n_recurrent: int, n_output: int):
        super().__init__()

        # Store architecture dimensions
        self.n_input = n_input
        self.n_recurrent = n_recurrent
        self.n_output = n_output

        # x2ah -> h2ah -> h2y
        self.fc_x2ah = nn.Linear(n_input, n_recurrent)
        self.fc_h2ah = nn.Linear(n_recurrent, n_recurrent, bias=False)
        self.fc_h2y = nn.Linear(n_recurrent, n_output)

        # Initial state ah0
        # Set requires_grad=False to exclude from optimization
        self.ah0 = nn.Parameter(torch.zeros(n_recurrent), requires_grad=False)

        self.tau = 10
        self.dt = 1

    def forward(self, x, activity_noise):
        """Forward pass of the CTRNN model.
        Args:
            x: Input tensor of shape (n_trials, n_T, n_input)
            activity_noise: Noise tensor
        """
        n_trials, n_T, _ = x.shape  # (n_trials, n_T, n_input)
        ah = self.ah0.repeat(n_trials, 1)

        h = retanh(ah)
        hstore = []
        for t in range(n_T):
            ah = ah + (self.dt / self.tau) * (-ah + self.fc_h2ah(h) +
                                              self.fc_x2ah(x[:, t]))
            h = retanh(ah) + activity_noise[:, t, :]
            hstore.append(h)
        hstore = torch.stack(hstore, dim=1)
        return self.fc_h2y(hstore), hstore

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
