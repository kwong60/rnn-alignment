from typing import Tuple
import numpy as np
import torch

from dataset import Trial


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

    mask = np.zeros((max_length, 2))
    mask[:seq_length, :] = 1

    return padded_input, padded_target, mask


def denormalize_positions(positions: torch.Tensor, position_mean: np.ndarray,
                          position_std: np.ndarray) -> torch.Tensor:
    return positions * torch.FloatTensor(position_std) + torch.FloatTensor(
        position_mean)
