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
        m = mask[i, :, 0] > 0

        h = h[m]
        y = y[m]

        # Center
        h0 = h - h.mean(dim=0, keepdim=True)
        y0 = y - y.mean(dim=0, keepdim=True)

        # small regularization for numerical stability
        eps = 1e-6
        h0 = h0 + eps * torch.randn_like(h0)
        y0 = y0 + eps * torch.randn_like(y0)

        # Procrustes without gradients
        with torch.no_grad():
            # Cross-covariance
            C = h0.T @ y0  # (H, N)
                
            # add regularization to C
            C = C + eps * torch.eye(C.shape[0], C.shape[1], device=C.device)
                
            # SVD
            U, S, Vt = torch.linalg.svd(C, full_matrices=False)
            V = Vt.T
                
            # rotation
            R = U @ V.T  # (H, N)
                
            # compute scale
            num = S.sum()
            denom = (h0 @ R).pow(2).sum().clamp(min=1e-8)
            scale = num / denom
            scale = scale.clamp(min=1e-3, max=1e3)  
                
        # apply transformation with gradients
        h_aligned = (h0 @ R.detach()) * scale.detach()

        # alignment loss
        loss = ((h_aligned - y0)**2).mean()
        losses.append(loss)

    return torch.stack(losses).mean()

def compute_alignment(hidden: torch.Tensor, neural: torch.Tensor, 
                              mask: torch.Tensor) -> dict:  
    with torch.no_grad():
        # first batch only
        h = hidden[0]  # (T, H)
        y = neural[0]  # (T, N)
        m = mask[0, :, 0] > 0
        
        h = h[m]
        y = y[m]
        
        if h.shape[0] < 2:
            return {'correlation': 0.0, 'mse_after_alignment': float('inf')}
        
        # Center
        h0 = h - h.mean(dim=0, keepdim=True)
        y0 = y - y.mean(dim=0, keepdim=True)
        
        # Procrustes alignment
        C = h0.T @ y0
        U, S, Vt = torch.linalg.svd(C, full_matrices=False)
        R = U @ Vt
        
        # Align
        h_aligned = h0 @ R

        # correlation between aligned representations
        correlation = torch.corrcoef(torch.cat([
            h_aligned.flatten().unsqueeze(0),
            y0.flatten().unsqueeze(0)
        ]))[0, 1].item()
        
        # MSE after alignment
        mse = ((h_aligned - y0)**2).mean().item()
        
        # explained variance
        var_y = y0.var()
        var_residual = (y0 - h_aligned).var()
        r_squared = 1 - (var_residual / var_y)
        r_squared = r_squared.item()
        
        return {
            'correlation': correlation,
            'mse_after_alignment': mse,
            'r_squared': r_squared,
        }

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

def process_neural_trial(trial: Trial, unit: int, max_length: int):
    start = trial.start_time
    end = trial.end_time
    bin_size = (end - start) / max_length
    
    bins = np.linspace(start, end, max_length + 1)
    neurons = [(c, u) for (c, u) in trial.spike_counts.keys() if u == unit and trial.spike_counts[(c, u)] > 0]
    neuron_len = len(neurons)
    
    num_bins = len(bins) - 1

    padded_fr_mat = np.zeros((neuron_len, num_bins))

    for j, (c, u) in enumerate(neurons):   
        times = trial.spike_times[(c, u)]
        # turn times into binned counts to add to matrix
        counts, _ = np.histogram(times, bins)
        # (#neurons x #max_bins)
        padded_fr_mat[j] = counts / bin_size

    # mask = np.zeros((len(neurons), max_bin_size))
    # mask[:, :bin_size] = 1
    # print(f'mat: {padded_fr_mat.shape}, bin_size {bin_size}, num_bins:{num_bins}')
    return padded_fr_mat


def denormalize_positions(positions: torch.Tensor, position_mean: np.ndarray,
                          position_std: np.ndarray) -> torch.Tensor:
    return positions * torch.FloatTensor(position_std) + torch.FloatTensor(
        position_mean)


