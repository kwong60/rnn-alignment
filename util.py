from typing import Tuple
import numpy as np
import torch

from dataset import Trial


def procrustes_loss(hidden: torch.Tensor, neural: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    hidden : (B, T, H)
    neural : (B, T, N)
    mask   : (B, T, 1)

    Computes rectangular generalized Procrustes alignment loss.
    """
    assert hidden.shape[0] == neural.shape[0] == mask.shape[0]
    B = hidden.shape[0]

    losses = []

    for i in range(B):
        h = hidden[i]  # (T, H)
        y = neural[i]  # (T, N)
        m = mask[i].squeeze() > 0

        h = h[m]
        y = y[m]

        # Center
        h0 = h - h.mean(dim=0, keepdim=True)
        y0 = y - y.mean(dim=0, keepdim=True)

        # Cross-covariance
        C = h0.T @ y0  # (H, N)

        # SVD
        U, S, Vt = torch.linalg.svd(C, full_matrices=False)
        V = Vt.T  # (N, k)

        # Correct rectangular Procrustes rotation
        R = U @ V.T  # (H, N)

        # Compute scale (generalized)
        # numerator = sum of singular values
        num = S.sum()
        # denominator = Frobenius norm of XR before scaling
        denom = (h0 @ R).pow(2).sum().clamp(min=1e-12)

        scale = num / denom  # better formulation

        # aligned hidden states
        h_aligned = (h0 @ R) * scale

        # alignment loss
        loss = ((h_aligned - y0)**2).mean()
        losses.append(loss)

    return torch.stack(losses).mean()


def process_trial(
        trial: Trial, position_mean: np.ndarray, position_std: np.ndarray,
        max_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cursor_positions = trial.cursor_positions.T  # (n_samples, 2)
    target_position: Tuple[float, float] = trial.target_position
    n_samples = cursor_positions.shape[0]

    # For each time t, we have [cursor_x[t], cursor_y[t], target_x, target_y]
    # Thus, the input sequence has shape (n_samples - 1, 4)
    seq_length = n_samples - 1
    input_seq = np.zeros((seq_length, 4))
    # Leave out the last cursor position
    input_seq[:, 0:2] = cursor_positions[:-1, :]
    input_seq[:, 2:4] = np.tile(target_position, (seq_length, 1))

    # Leave out the first cursor position as target
    target_seq = cursor_positions[1:, :]

    # Normalization
    input_seq[:, 0:2] = (input_seq[:, 0:2] - position_mean) / position_std
    input_seq[:, 2:4] = (input_seq[:, 2:4] - position_mean) / position_std
    target_seq = (target_seq - position_mean) / position_std

    padded_input = np.zeros((max_length, 4))
    padded_input[:seq_length, :] = input_seq

    padded_target = np.zeros((max_length, 2))
    padded_target[:seq_length, :] = target_seq

    mask = np.zeros((max_length, 1))
    mask[:seq_length, :] = 1

    return padded_input, padded_target, mask


def denormalize_positions(positions: torch.Tensor, position_mean: np.ndarray,
                          position_std: np.ndarray) -> torch.Tensor:
    return positions * torch.FloatTensor(position_std) + torch.FloatTensor(
        position_mean)
