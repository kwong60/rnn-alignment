from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import Trial
from model import CTRNN
import util


def training_curves(save_path: Path, train_losses: list[float],
                    val_losses: list[float],
                    val_losses_denorm: list[float]) -> None:

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs,
             val_losses_denorm,
             label='Validation Loss (Denorm)',
             color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Denormalized Loss')
    plt.title('Validation Loss (Denormalized)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def trial_prediction(save_path: Path, model: CTRNN, trial: Trial,
                     position_mean: np.ndarray, position_std: np.ndarray,
                     max_length: int) -> None:

    model.eval()

    with torch.no_grad():
        # Prepare padded input, target, and masks
        padded_input, padded_target, mask = util.process_trial(
            trial, position_mean, position_std, max_length)

        valid_length = int(mask[:, 0].sum())

        input_tensor = torch.FloatTensor(padded_input).unsqueeze(
            0)  # (1, T, 4)
        n_T = input_tensor.shape[1]

        noise = torch.zeros(1, n_T, model.n_recurrent)

        outputs, _ = model(input_tensor, noise)  # (1, T, 2)

    # ---- Denormalize ----
    outputs_denorm = util.denormalize_positions(outputs, position_mean,
                                                position_std)[0]  # (T, 2)
    target_denorm = util.denormalize_positions(
        torch.FloatTensor(padded_target), position_mean,
        position_std)  # (T, 2)

    # True target position is stored in input[:, 2:4].
    target_pos_denorm = util.denormalize_positions(
        torch.FloatTensor(padded_input[:, 2:4]), position_mean,
        position_std)  # (T, 2)

    target_xy = target_pos_denorm[0].numpy()  # constant across time

    # Convert arrays for plotting
    outputs_np = outputs_denorm.numpy()
    targets_np = target_denorm.numpy()

    # ---- Extract valid timesteps ----
    outputs_np = outputs_np[:valid_length]
    targets_np = targets_np[:valid_length]

    # ---- Create plot ----
    plt.figure(figsize=(10, 8))

    # Ground truth trajectory
    plt.plot(targets_np[:, 0],
             targets_np[:, 1],
             'b-',
             linewidth=2,
             label="Ground Truth",
             alpha=0.7)

    # Predicted trajectory
    plt.plot(outputs_np[:, 0],
             outputs_np[:, 1],
             'r--',
             linewidth=2,
             label="Predicted",
             alpha=0.7)

    # Target position
    plt.scatter(target_xy[0],
                target_xy[1],
                s=200,
                c='green',
                marker='*',
                label='Target',
                zorder=10,
                edgecolors='black')

    # Start point (GT)
    plt.scatter(targets_np[0, 0],
                targets_np[0, 1],
                s=100,
                c='blue',
                marker='o',
                label='Start',
                zorder=10,
                edgecolors='black')

    # End points
    plt.scatter(targets_np[-1, 0],
                targets_np[-1, 1],
                s=100,
                c='blue',
                marker='s',
                label='GT End',
                zorder=10,
                edgecolors='black')

    plt.scatter(outputs_np[-1, 0],
                outputs_np[-1, 1],
                s=100,
                c='red',
                marker='s',
                label='Pred End',
                zorder=10,
                edgecolors='black')

    plt.xlabel("X Position (mm)", fontsize=12)
    plt.ylabel("Y Position (mm)", fontsize=12)
    plt.title(f"Cursor Trajectory Prediction (Trial {trial.trial_number})",
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
