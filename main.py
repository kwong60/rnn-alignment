from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

from dataset import Dataset
from model import CTRNN
from typing import Tuple
import util
import plot

N_INPUT = 4  # [cursor_x, cursor_y, target_x, target_y]
N_RECURRENT = 128  # Hidden units
N_OUTPUT = 2  # [next_cursor_x, next_cursor_y]

N_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TRAIN_RATIO = 0.8

DATA_PATH = "data/indy_20160407_02.mat"
PLOT_DIR = "plots/"


class ProcessedData:
    inputs: torch.FloatTensor
    targets: torch.FloatTensor
    masks: torch.FloatTensor

    position_mean: np.ndarray
    position_std: np.ndarray
    max_length: int

    def __init__(self, dataset: Dataset, min_length: int = 10):
        # Fetch all trials
        trials = [
            dataset.get_trial(i) for i in range(dataset.number_of_trials)
        ]
        # Filter trials based on minimum length
        trials = [
            t for t in trials if t.cursor_positions.shape[1] >= min_length
        ]

        max_length = max(trial.cursor_positions.shape[1]
                         for trial in trials) - 1

        assert max_length >= min_length
        assert max_length > 0

        # Compute normalization statistics
        all_positions = np.concatenate(
            [trial.cursor_positions.T for trial in trials], axis=0)
        position_mean = all_positions.mean(axis=0)  # (2,)
        position_std = all_positions.std(axis=0)  # (2,)
        eps = 1e-6
        position_std = np.maximum(position_std, eps)

        inputs_list = []
        targets_list = []
        masks_list = []

        for trial in trials:
            padded_input, padded_target, mask = util.process_trial(
                trial, position_mean, position_std, max_length)
            inputs_list.append(padded_input)
            targets_list.append(padded_target)
            masks_list.append(mask)

        self.inputs = torch.FloatTensor(np.stack(inputs_list, axis=0))
        self.targets = torch.FloatTensor(np.stack(targets_list, axis=0))
        self.masks = torch.FloatTensor(np.stack(masks_list, axis=0))
        self.position_mean = position_mean
        self.position_std = position_std
        self.max_length = max_length

    def split_train_val(
            self, train_ratio: float) -> Tuple[TensorDataset, TensorDataset]:
        n_trials = self.inputs.shape[0]
        n_train = int(n_trials * train_ratio)

        indices = np.random.permutation(n_trials)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_inputs = torch.FloatTensor(self.inputs[train_indices])
        train_targets = torch.FloatTensor(self.targets[train_indices])
        train_masks = torch.FloatTensor(self.masks[train_indices])

        val_inputs = torch.FloatTensor(self.inputs[val_indices])
        val_targets = torch.FloatTensor(self.targets[val_indices])
        val_masks = torch.FloatTensor(self.masks[val_indices])

        return (TensorDataset(train_inputs, train_targets, train_masks),
                TensorDataset(val_inputs, val_targets, val_masks))


def masked_mse_loss(predictions, targets, masks):
    # Element-wise squared error
    squared_error = (predictions - targets)**2
    # Apply mask
    masked_error = squared_error * masks
    # Sum over all dimensions and divide by number of valid elements
    total_error = masked_error.sum()
    num_valid = masks.sum()
    # Safeguard against division by zero
    assert num_valid > 0
    return total_error / num_valid


def train_epoch(model: CTRNN, optimizer: torch.optim.Optimizer,
                train_loader: DataLoader) -> tuple[float, torch.Tensor]:
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    all_hidden_states = []

    for batch_inputs, batch_targets, batch_masks in train_loader:
        batch_size_actual = batch_inputs.shape[0]
        max_length = batch_inputs.shape[1]

        # NOTE: Zero noise being added...
        noise = torch.zeros(batch_size_actual, max_length, model.n_recurrent)

        outputs, hidden_states = model(batch_inputs, noise)

        all_hidden_states.append(hidden_states.detach())

        # TODO: Do something with hidden_states?
        assert torch.isfinite(outputs).all()

        loss = masked_mse_loss(outputs, batch_targets, batch_masks)
        assert torch.isfinite(loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    all_hidden_states = torch.cat(all_hidden_states, dim=0)  # (n_train_trials, n_T, n_recurrent)
    
    return epoch_loss / num_batches, all_hidden_states


def denormalize_positions(positions: torch.Tensor, position_mean: np.ndarray,
                          position_std: np.ndarray) -> torch.Tensor:
    return positions * torch.FloatTensor(position_std) + torch.FloatTensor(
        position_mean)


def evaluate(model: CTRNN, val_loader: TensorDataset,
             position_mean: np.ndarray, position_std: np.ndarray) -> dict:
    model.eval()

    with torch.no_grad():
        inputs, targets, masks = val_loader.tensors

        n_trials = inputs.shape[0]
        max_length = inputs.shape[1]
        n_recurrent = model.n_recurrent

        # NOTE: Zero noise being added...
        noise = torch.zeros(n_trials, max_length, n_recurrent)

        outputs, _ = model(inputs, noise)
        assert torch.isfinite(outputs).all()

        loss = masked_mse_loss(outputs, targets, masks).item()

        outputs_denorm = denormalize_positions(outputs, position_mean,
                                               position_std)
        targets_denorm = denormalize_positions(targets, position_mean,
                                               position_std)
        loss_denorm = masked_mse_loss(outputs_denorm, targets_denorm,
                                      masks).item()

    return {
        "loss": loss,
        "loss_denorm": loss_denorm,
    }


if __name__ == "__main__":
    dataset = Dataset(DATA_PATH)
    processed_data = ProcessedData(dataset)
    train_data, val_data = processed_data.split_train_val(
        train_ratio=TRAIN_RATIO)
    model = CTRNN(n_input=N_INPUT, n_recurrent=N_RECURRENT, n_output=N_OUTPUT)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_losses_denorm = []

    weight_history = []
    hidden_state_history = []

    SAVE_INTERVAL = 1
    
    eval_inputs, eval_targets, eval_masks = val_data.tensors

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        # TRAINING
        train_loss, hidden_states = train_epoch(model, optimizer, train_loader)

        assert np.isfinite(train_loss)
        train_losses.append(train_loss)

        # EVALUATION
        val_metrics = evaluate(
            model,
            val_data,
            position_mean=processed_data.position_mean,
            position_std=processed_data.position_std,
        )
        val_loss = val_metrics["loss"]
        val_loss_denorm = val_metrics["loss_denorm"]
        val_losses.append(val_loss)
        val_losses_denorm.append(val_loss_denorm)
        print(
            f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Loss (denorm): {val_loss_denorm:.6f}"
        )

        # save weights every SAVE_INTERVAL epochs
        if epoch % SAVE_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                max_length = eval_inputs.shape[1]
                noise = torch.zeros(len(eval_inputs), max_length, N_RECURRENT)
                # weight_history.append({
                #     'epoch': epoch,
                #     'weights': model.get_weight_snapshot(),
                #     'train_loss': train_loss,
                #     'val_loss': val_loss,
                # })

                _, hidden_states = model(eval_inputs, noise)
        
                print(f"\nHidden states at epoch {epoch}:")
                print(f"  Shape: {hidden_states.shape}")
                print(f"  Min: {hidden_states.min().item():.4f}")
                print(f"  Max: {hidden_states.max().item():.4f}")
                print(f"  Mean: {hidden_states.mean().item():.4f}")
          
                print(f"  First 5 units:")
                print(f"  {hidden_states[1, 1, :5]}")
     
                hidden_state_history.append({
                    'epoch': epoch,
                    'hidden_states': hidden_states.clone(),  # (eval_batch_size, n_T, n_recurrent)
                    'val_loss': val_loss,
                })
        
        model.train()

    # PLOTTING
    # Create plot directory if it doesn't exist
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

    plot.training_curves(
        Path(PLOT_DIR) / "training_curves.png", train_losses, val_losses,
        val_losses_denorm)
    # Plot prediction on a random validation trial
    random_trial = dataset.get_trial(180)
    plot.trial_prediction(
        Path(PLOT_DIR) / "trial_prediction.png", model, random_trial,
        processed_data.position_mean, processed_data.position_std,
        processed_data.max_length)
