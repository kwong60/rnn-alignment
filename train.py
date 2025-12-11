"""
Training script for cursor position prediction using CTRNN.
Trains model to predict cursor trajectory given initial position and target.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset
from model import CTRNN

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

# Model parameters
N_INPUT = 4  # [cursor_x, cursor_y, target_x, target_y]
N_RECURRENT = 128  # Hidden units
N_OUTPUT = 2  # [next_cursor_x, next_cursor_y]

# Training parameters
LEARNING_RATE = 1e-3
N_EPOCHS = 25
BATCH_SIZE = 32
GRADIENT_CLIP_NORM = 10.0
TRAIN_RATIO = 0.8

# Data parameters
DATA_PATH = "data/indy_20160407_02.mat"
CHECKPOINT_DIR = "checkpoints/"
PLOT_DIR = "plots/"

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# SECTION 2: DATA PREPARATION FUNCTIONS
# ============================================================================


def load_and_prepare_data(dataset, normalize=True, min_length=10):
    """
    Load trials and prepare padded sequences with masks.

    Args:
        dataset: Dataset object
        normalize: Whether to normalize position data
        min_length: Minimum number of samples required per trial (default: 10)

    Returns:
        inputs: (n_trials, max_length, 4) - input sequences
        targets: (n_trials, max_length, 2) - target sequences
        masks: (n_trials, max_length, 2) - validity masks
        position_mean: (2,) - mean for each position dimension
        position_std: (2,) - std for each position dimension
    """
    print("Loading and preparing data...")

    # Extract all trials
    n_trials = dataset.number_of_trials
    all_trials = [dataset.get_trial(i) for i in range(n_trials)]

    # Filter out trials that are too short
    trial_lengths = [t.cursor_positions.shape[1] for t in all_trials]
    trials = [
        t for t in all_trials if t.cursor_positions.shape[1] >= min_length
    ]
    n_filtered = n_trials - len(trials)

    print(
        f"Loaded {n_trials} trials, filtered out {n_filtered} trials with < {min_length} samples"
    )
    print(
        f"Trial length stats: min={min(trial_lengths)}, max={max(trial_lengths)}, "
        f"mean={np.mean(trial_lengths):.1f}")
    print(f"Using {len(trials)} trials for training")

    # Find max sequence length
    max_length = max(trial.cursor_positions.shape[1] - 1
                     for trial in trials)  # -1 for shifting
    print(f"Max sequence length: {max_length}")

    # Collect all cursor positions for normalization statistics
    all_positions = []
    for trial in trials:
        # Transpose from (2, n_samples) to (n_samples, 2)
        positions = trial.cursor_positions.T  # (n_samples, 2)
        all_positions.append(positions)
    all_positions = np.concatenate(all_positions, axis=0)  # (total_samples, 2)

    # Compute normalization statistics
    position_mean = all_positions.mean(axis=0)  # (2,)
    position_std = all_positions.std(axis=0)  # (2,)

    # Add epsilon to prevent division by zero or very small numbers
    eps = 1e-6
    position_std = np.maximum(position_std, eps)

    print(f"Position mean: {position_mean}")
    print(f"Position std: {position_std}")

    # Prepare sequences for each trial
    inputs_list = []
    targets_list = []
    masks_list = []

    for trial in trials:
        # Get cursor positions and target position
        cursor_pos = trial.cursor_positions.T  # (n_samples, 2)
        target_pos = np.array(trial.target_position)  # (2,)
        n_samples = cursor_pos.shape[0]

        # Create input sequence: [cursor_x[t], cursor_y[t], target_x, target_y]
        # Use timesteps [0, 1, 2, ..., n-2]
        seq_length = n_samples - 1
        input_seq = np.zeros((seq_length, 4))
        input_seq[:, 0:2] = cursor_pos[:-1, :]  # Current cursor position
        input_seq[:,
                  2:4] = np.tile(target_pos,
                                 (seq_length, 1))  # Target position (constant)

        # Create target sequence: [cursor_x[t+1], cursor_y[t+1]]
        # Use timesteps [1, 2, 3, ..., n-1]
        target_seq = cursor_pos[1:, :]  # (seq_length, 2)

        # Normalize if requested
        if normalize:
            input_seq[:,
                      0:2] = (input_seq[:, 0:2] - position_mean) / position_std
            input_seq[:,
                      2:4] = (input_seq[:, 2:4] - position_mean) / position_std
            target_seq = (target_seq - position_mean) / position_std

        # Pad to max_length
        padded_input = np.zeros((max_length, 4))
        padded_input[:seq_length, :] = input_seq

        padded_target = np.zeros((max_length, 2))
        padded_target[:seq_length, :] = target_seq

        # Create mask (1 for valid, 0 for padding)
        mask = np.zeros((max_length, 2))
        mask[:seq_length, :] = 1

        inputs_list.append(padded_input)
        targets_list.append(padded_target)
        masks_list.append(mask)

    # Stack into arrays
    inputs = np.stack(inputs_list, axis=0)  # (n_trials, max_length, 4)
    targets = np.stack(targets_list, axis=0)  # (n_trials, max_length, 2)
    masks = np.stack(masks_list, axis=0)  # (n_trials, max_length, 2)

    print(
        f"Data shapes - inputs: {inputs.shape}, targets: {targets.shape}, masks: {masks.shape}"
    )

    return inputs, targets, masks, position_mean, position_std


def create_train_val_split(inputs, targets, masks, train_ratio=0.8, seed=None):
    """
    Split data into training and validation sets.

    Args:
        inputs: (n_trials, max_length, 4)
        targets: (n_trials, max_length, 2)
        masks: (n_trials, max_length, 2)
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_inputs, train_targets, train_masks,
                  val_inputs, val_targets, val_masks)
    """
    if seed is not None:
        np.random.seed(seed)

    n_trials = inputs.shape[0]
    n_train = int(train_ratio * n_trials)

    # Random permutation
    indices = np.random.permutation(n_trials)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"Train/Val split: {len(train_indices)}/{len(val_indices)} trials")

    # Split data
    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    train_masks = masks[train_indices]

    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]
    val_masks = masks[val_indices]

    return (train_inputs, train_targets, train_masks, val_inputs, val_targets,
            val_masks)


def denormalize_positions(positions, mean, std):
    """Denormalize position data back to original scale."""
    return positions * std + mean


# ============================================================================
# SECTION 3: LOSS FUNCTION
# ============================================================================


def masked_mse_loss(predictions, targets, masks):
    """
    Compute MSE loss only on non-padded positions.

    Args:
        predictions: (batch_size, n_T, n_output)
        targets: (batch_size, n_T, n_output)
        masks: (batch_size, n_T, n_output) - 1 for valid, 0 for padding

    Returns:
        Scalar loss value
    """
    # Element-wise squared error
    squared_error = (predictions - targets)**2

    # Apply mask
    masked_error = squared_error * masks

    # Sum over all dimensions and divide by number of valid elements
    total_error = masked_error.sum()
    num_valid = masks.sum()

    # Safeguard against division by zero
    if num_valid == 0:
        print("WARNING: No valid elements in batch (all masked)!")
        return torch.tensor(0.0, requires_grad=True)

    return total_error / num_valid


# ============================================================================
# SECTION 4: TRAINING AND EVALUATION FUNCTIONS
# ============================================================================


def train_epoch(model, optimizer, train_loader, n_recurrent):
    """
    Train for one epoch.

    Args:
        model: CTRNN model
        optimizer: Optimizer
        train_loader: DataLoader for training data
        n_recurrent: Number of recurrent units (for noise tensor)

    Returns:
        Tuple of (average loss, max gradient norm)
    """
    model.train()

    epoch_loss = 0.0
    num_batches = 0
    max_grad_norm = 0.0

    # Process batches from DataLoader
    for batch_inputs, batch_targets, batch_masks in train_loader:
        batch_size_actual = batch_inputs.shape[0]
        max_length = batch_inputs.shape[1]

        # Create activity noise (zeros for now)
        # Activity noise set to zero for now; can be replaced with Gaussian noise
        # if we want to regularize or match neural variability.
        noise = torch.zeros(batch_size_actual, max_length, n_recurrent)

        # Forward pass
        outputs, hidden_states = model(batch_inputs, noise)

        # Check for inf/nan in outputs before computing loss
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("WARNING: NaN/Inf in model outputs!")
            print(
                f"  Output stats: min={outputs.min().item():.3f}, max={outputs.max().item():.3f}, "
                f"mean={outputs.mean().item():.3f}")
            print(
                f"  Hidden stats: min={hidden_states.min().item():.3f}, max={hidden_states.max().item():.3f}"
            )
            return float('nan'), float('nan')

        # Compute loss
        loss = masked_mse_loss(outputs, batch_targets, batch_masks)

        # Check for NaN in loss
        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            print(
                f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]"
            )
            print(
                f"  Target range: [{batch_targets.min().item():.3f}, {batch_targets.max().item():.3f}]"
            )
            return float('nan'), float('nan')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   GRADIENT_CLIP_NORM)
        max_grad_norm = max(max_grad_norm, grad_norm.item())

        # Update
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches, max_grad_norm


def evaluate(model, inputs, targets, masks, position_mean, position_std):
    """
    Evaluate model on validation set.

    Args:
        model: CTRNN model
        inputs: (n_trials, max_length, 4)
        targets: (n_trials, max_length, 2)
        masks: (n_trials, max_length, 2)
        position_mean: (2,) - for denormalization
        position_std: (2,) - for denormalization

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        n_trials = inputs.shape[0]
        max_length = inputs.shape[1]
        n_recurrent = model.n_recurrent

        # Create activity noise (zeros)
        noise = torch.zeros(n_trials, max_length, n_recurrent)

        # Forward pass
        outputs, _ = model(inputs, noise)

        # Compute normalized MSE
        normalized_mse = masked_mse_loss(outputs, targets, masks).item()

        # Compute denormalized MSE for interpretability
        outputs_denorm = denormalize_positions(outputs, position_mean,
                                               position_std)
        targets_denorm = denormalize_positions(targets, position_mean,
                                               position_std)
        denormalized_mse = masked_mse_loss(outputs_denorm, targets_denorm,
                                           masks).item()

    return {
        'normalized_mse': normalized_mse,
        'denormalized_mse': denormalized_mse,
    }


# ============================================================================
# SECTION 5: VISUALIZATION FUNCTIONS
# ============================================================================


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Normalized MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_predictions(model, inputs, targets, masks, trial_idx, position_mean,
                     position_std, save_path):
    """
    Visualize predictions for a single trial.

    Args:
        model: CTRNN model
        inputs: (n_trials, max_length, 4)
        targets: (n_trials, max_length, 2)
        masks: (n_trials, max_length, 2)
        trial_idx: Index of trial to visualize
        position_mean: (2,) - for denormalization
        position_std: (2,) - for denormalization
        save_path: Path to save figure
    """
    model.eval()

    with torch.no_grad():
        # Get single trial
        trial_input = inputs[trial_idx:trial_idx + 1]  # (1, max_length, 4)
        trial_target = targets[trial_idx:trial_idx + 1]  # (1, max_length, 2)
        trial_mask = masks[trial_idx:trial_idx + 1]  # (1, max_length, 2)

        max_length = trial_input.shape[1]
        n_recurrent = model.n_recurrent

        # Create noise
        noise = torch.zeros(1, max_length, n_recurrent)

        # Forward pass
        output, _ = model(trial_input, noise)

        # Denormalize before converting to numpy (more efficient)
        output_denorm = denormalize_positions(output[0], position_mean,
                                              position_std).cpu().numpy()
        target_denorm = denormalize_positions(trial_target[0], position_mean,
                                              position_std).cpu().numpy()
        target_pos_denorm = denormalize_positions(trial_input[0, :, 2:4],
                                                  position_mean,
                                                  position_std).cpu().numpy()

        # Convert mask to numpy
        mask_np = trial_mask[0].cpu().numpy()  # (max_length, 2)

        # Find valid length (where mask is 1)
        valid_length = int(mask_np[:, 0].sum())

        # Extract target position (should be constant)
        target_position = target_pos_denorm[0]  # (2,)

        # Plot
        plt.figure(figsize=(10, 8))

        # Plot ground truth trajectory
        plt.plot(target_denorm[:valid_length, 0],
                 target_denorm[:valid_length, 1],
                 'b-',
                 linewidth=2,
                 label='Ground Truth',
                 alpha=0.7)

        # Plot predicted trajectory
        plt.plot(output_denorm[:valid_length, 0],
                 output_denorm[:valid_length, 1],
                 'r--',
                 linewidth=2,
                 label='Predicted',
                 alpha=0.7)

        # Plot target position
        plt.scatter(target_position[0],
                    target_position[1],
                    s=200,
                    c='green',
                    marker='*',
                    label='Target',
                    zorder=10,
                    edgecolors='black')

        # Plot start position
        plt.scatter(target_denorm[0, 0],
                    target_denorm[0, 1],
                    s=100,
                    c='blue',
                    marker='o',
                    label='Start',
                    zorder=10,
                    edgecolors='black')

        # Plot end positions
        plt.scatter(target_denorm[valid_length - 1, 0],
                    target_denorm[valid_length - 1, 1],
                    s=100,
                    c='blue',
                    marker='s',
                    label='GT End',
                    zorder=10,
                    edgecolors='black')
        plt.scatter(output_denorm[valid_length - 1, 0],
                    output_denorm[valid_length - 1, 1],
                    s=100,
                    c='red',
                    marker='s',
                    label='Pred End',
                    zorder=10,
                    edgecolors='black')

        plt.xlabel('X Position (mm)', fontsize=12)
        plt.ylabel('Y Position (mm)', fontsize=12)
        plt.title(f'Cursor Trajectory - Trial {trial_idx}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved prediction plot to {save_path}")


# ============================================================================
# SECTION 6: MAIN TRAINING SCRIPT
# ============================================================================


def main():
    """Main training function."""
    print("=" * 80)
    print("CTRNN Cursor Position Prediction Training")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create directories
    Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(PLOT_DIR).mkdir(exist_ok=True)

    # Load dataset
    dataset = Dataset(DATA_PATH)

    # Prepare data
    inputs, targets, masks, position_mean, position_std = load_and_prepare_data(
        dataset, normalize=True)

    # Convert to tensors
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets)
    masks = torch.FloatTensor(masks)
    position_mean = torch.FloatTensor(position_mean)
    position_std = torch.FloatTensor(position_std)

    # Train/val split
    (train_inputs, train_targets, train_masks, val_inputs, val_targets,
     val_masks) = create_train_val_split(inputs,
                                         targets,
                                         masks,
                                         train_ratio=TRAIN_RATIO,
                                         seed=RANDOM_SEED)

    # Create DataLoaders
    train_dataset = TensorDataset(train_inputs, train_targets, train_masks)
    val_dataset = TensorDataset(val_inputs, val_targets, val_masks)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\nInitializing CTRNN model...")
    print(f"  n_input: {N_INPUT}")
    print(f"  n_recurrent: {N_RECURRENT}")
    print(f"  n_output: {N_OUTPUT}")
    model = CTRNN(n_input=N_INPUT, n_recurrent=N_RECURRENT, n_output=N_OUTPUT)
    print(f"  Total parameters: {model.n_parameters}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nStarting training for {N_EPOCHS} epochs...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient clip norm: {GRADIENT_CLIP_NORM}")
    print("=" * 80)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    weight_history = []

    SAVE_INTERVAL = 5 
    
    eval_inputs = val_inputs[:len(val_inputs)]
    eval_targets = val_targets[:len(val_inputs)]
    eval_masks = val_masks[:len(val_inputs)]

    for epoch in range(N_EPOCHS):
        # Train
        train_loss, max_grad_norm = train_epoch(model, optimizer, train_loader,
                                                N_RECURRENT)
        train_losses.append(train_loss)

        # Check for NaN
        if np.isnan(train_loss):
            print(f"\nTraining stopped at epoch {epoch} due to NaN loss!")
            break

        # Evaluate
        val_metrics = evaluate(model, val_inputs, val_targets, val_masks,
                               position_mean, position_std)
        val_loss = val_metrics['normalized_mse']
        val_rmse_denorm = np.sqrt(val_metrics['denormalized_mse'])
        val_losses.append(val_loss)

        # save weights every SAVE_INTERVAL epochs
        if epoch % SAVE_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                max_length = eval_inputs.shape[1]
                noise = torch.zeros(len(val_inputs), max_length, N_RECURRENT)
                weight_history.append({
                    'epoch': epoch,
                    'weights': model.get_weight_snapshot(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                })
            
            model.train()

        # Log progress
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val RMSE: {val_rmse_denorm:.2f} mm | "
                  f"Grad Norm: {max_grad_norm:.2f}")

        # Save checkpoints
        if epoch % 50 == 0 and epoch > 0:
            checkpoint_path = Path(CHECKPOINT_DIR) / f"model_epoch_{epoch}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'position_mean': position_mean,
                'position_std': position_std,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = Path(CHECKPOINT_DIR) / "model_best.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'position_mean': position_mean,
                'position_std': position_std,
            }
            torch.save(checkpoint, best_model_path)

    print("=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    final_model_path = Path(CHECKPOINT_DIR) / "model_final.pth"
    checkpoint = {
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'position_mean': position_mean,
        'position_std': position_std,
    }
    torch.save(checkpoint, final_model_path)
    print(f"Saved final model to {final_model_path}")

    history_path = Path(CHECKPOINT_DIR) / "training_history.pth"
    history_data = {
        'weight_history': weight_history,
        'eval_inputs': eval_inputs,
        'eval_targets': eval_targets,
        'eval_masks': eval_masks,
        'position_mean': position_mean,
        'position_std': position_std,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(history_data, history_path)
    print(f"Saved training history to {history_path}")

    # Plot training curves
    plot_path = Path(PLOT_DIR) / "training_curves.png"
    plot_training_curves(train_losses, val_losses, plot_path)

    # Visualize predictions on sample validation trials
    print("\nGenerating prediction visualizations...")
    for i in range(min(3, len(val_inputs))):
        plot_path = Path(PLOT_DIR) / f"prediction_trial_{i}.png"
        plot_predictions(model, val_inputs, val_targets, val_masks, i,
                         position_mean, position_std, plot_path)

    print("=" * 80)
    print("All done!")


if __name__ == '__main__':
    main()