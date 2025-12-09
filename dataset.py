"""
Dataset module for loading and visualizing neural reaching data.

This module provides a convenient interface for working with neural recording
data from reaching experiments, including trial segmentation, spike extraction,
and visualization.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path


@dataclass
class Trial:
    """
    Represents a single trial from the reaching experiment.

    Attributes:
        trial_number: Index of the trial
        start_time: Start time in seconds
        end_time: End time in seconds
        duration: Trial duration in seconds
        target_position: Target (x, y) position in mm
        cursor_positions: Array of cursor (x, y) positions over time, shape (2, n_samples)
        finger_positions: Array of finger (z, -x, -y) positions over time, shape (3, n_samples)
        timestamps: Array of timestamps for this trial, shape (n_samples,)
        spike_times: Dict mapping (channel, unit) to array of spike times
        spike_counts: Dict mapping (channel, unit) to total spike count
    """
    trial_number: int
    start_time: float
    end_time: float
    duration: float
    target_position: Tuple[float, float]
    cursor_positions: np.ndarray
    finger_positions: np.ndarray
    timestamps: np.ndarray
    spike_times: Dict[Tuple[int, int], np.ndarray]
    spike_counts: Dict[Tuple[int, int], int]

    def __repr__(self):
        return (
            f"Trial(number={self.trial_number}, duration={self.duration:.2f}s, "
            f"target={self.target_position}, neurons={len(self.spike_times)})")


class Dataset:
    """
    Loads and manages neural reaching data from HDF5/MATLAB files.

    This class handles loading, segmentation, and visualization of neural
    recordings from a reaching experiment. Data is stored in memory for
    efficient access.

    Example:
        >>> dataset = Dataset("data/indy_20160407_02.mat")
        >>> print(dataset)
        >>> trial = dataset.get_trial(0)
        >>> dataset.visualize_trial_position(0)
    """

    def __init__(self, filepath: str):
        """
        Initialize the dataset from a .mat file.

        Args:
            filepath: Path to the HDF5/MATLAB file containing the data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading dataset from {self.filepath}...")

        # Open file and keep it open for the lifetime of the object
        self._file = h5py.File(str(self.filepath), 'r')

        # Load continuous data
        self._load_continuous_data()

        # Segment into trials
        self._segment_trials()

        # Load channel names
        self._load_channel_names()

        print(f"Dataset loaded: {self.number_of_trials} trials, "
              f"{self.number_of_channels} channels, {self.duration:.2f}s")

    def _load_continuous_data(self):
        """Load all continuous behavioral data into memory."""
        self.timestamps = self._file['t'][:].flatten()
        self.cursor_positions = self._file['cursor_pos'][:]  # (2, n_samples)
        self.finger_positions = self._file['finger_pos'][:]  # (3, n_samples)
        self.target_positions = self._file['target_pos'][:]  # (2, n_samples)

        # Store spike and waveform references (not loaded into memory yet)
        self.spikes_ref = self._file['spikes']
        self.waveforms_ref = self._file['wf']

        # Derived properties
        self.number_of_channels = self.spikes_ref.shape[1]
        self.number_of_units = self.spikes_ref.shape[0]
        self.duration = self.timestamps[-1] - self.timestamps[0]
        self.sampling_rate = 1.0 / np.median(np.diff(self.timestamps))

    def _load_channel_names(self):
        """Load and decode channel names."""
        chan_names_ref = self._file['chan_names']
        self.channel_names = []

        for i in range(chan_names_ref.shape[1]):
            chan_ref = chan_names_ref[0, i]
            if chan_ref:
                try:
                    name_chars = self._file[chan_ref][:].flatten()
                    name_str = ''.join(
                        chr(int(c)) for c in name_chars if c > 0)
                    self.channel_names.append(name_str)
                except:
                    self.channel_names.append(f"Ch{i}")
            else:
                self.channel_names.append(f"Ch{i}")

    def _segment_trials(self):
        """Segment continuous data into trials based on target position changes."""
        # Detect when target position changes
        target_changes = ((np.diff(self.target_positions[0, :]) != 0) |
                          (np.diff(self.target_positions[1, :]) != 0))
        trial_boundaries_idx = np.where(target_changes)[0] + 1

        # Create trial start/end indices
        self.trial_starts = np.concatenate([[0], trial_boundaries_idx])
        self.trial_ends = np.concatenate(
            [trial_boundaries_idx, [len(self.timestamps)]])

    @property
    def number_of_trials(self) -> int:
        """Return the total number of trials."""
        return len(self.trial_starts)

    def get_trial(self, trial_number: int) -> Trial:
        """
        Get data for a specific trial.

        Args:
            trial_number: Index of the trial (0-indexed)

        Returns:
            Trial object containing all trial data

        Raises:
            ValueError: If trial_number is out of range
        """
        if trial_number < 0 or trial_number >= self.number_of_trials:
            raise ValueError(f"Trial number {trial_number} out of range "
                             f"[0, {self.number_of_trials - 1}]")

        # Get trial indices
        start_idx = self.trial_starts[trial_number]
        end_idx = self.trial_ends[trial_number]

        # Get time range
        start_time = self.timestamps[start_idx]
        end_time = self.timestamps[end_idx - 1]
        duration = end_time - start_time

        # Extract behavioral data
        trial_cursor = self.cursor_positions[:, start_idx:end_idx]
        trial_finger = self.finger_positions[:, start_idx:end_idx]
        trial_timestamps = self.timestamps[start_idx:end_idx]
        trial_target = (self.target_positions[0, start_idx],
                        self.target_positions[1, start_idx])

        # Extract neural data
        spike_times_dict = {}
        spike_counts_dict = {}

        for channel in range(self.number_of_channels):
            for unit in range(self.number_of_units):
                spike_ref = self.spikes_ref[unit, channel]
                if spike_ref:
                    spike_data = self._file[spike_ref]
                    if spike_data.size > 2:  # Not empty reference
                        all_spike_times = spike_data[:].flatten()

                        # Filter spikes within trial time range
                        trial_spikes = all_spike_times[
                            (all_spike_times >= start_time)
                            & (all_spike_times <= end_time)]

                        if len(trial_spikes) > 0:
                            spike_times_dict[(channel, unit)] = trial_spikes
                            spike_counts_dict[(channel,
                                               unit)] = len(trial_spikes)

        return Trial(
            trial_number=trial_number,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            target_position=trial_target,
            cursor_positions=trial_cursor,
            finger_positions=trial_finger,
            timestamps=trial_timestamps -
            start_time,  # Relative to trial start
            spike_times=spike_times_dict,
            spike_counts=spike_counts_dict)

    def get_spike_times(self, channel: int, unit: int) -> np.ndarray:
        """
        Get all spike times for a specific channel and unit.

        Args:
            channel: Channel index (0 to number_of_channels-1)
            unit: Unit index (0=unsorted, 1=sorted1, 2=sorted2, etc.)

        Returns:
            Array of spike times in seconds

        Raises:
            ValueError: If channel or unit is out of range
        """
        if channel < 0 or channel >= self.number_of_channels:
            raise ValueError(f"Channel {channel} out of range")
        if unit < 0 or unit >= self.number_of_units:
            raise ValueError(f"Unit {unit} out of range")

        spike_ref = self.spikes_ref[unit, channel]
        if not spike_ref:
            return np.array([])

        spike_data = self._file[spike_ref]
        if spike_data.size <= 2:
            return np.array([])

        return spike_data[:].flatten()

    def get_waveforms(self, channel: int, unit: int) -> Optional[np.ndarray]:
        """
        Get waveforms for a specific channel and unit.

        Args:
            channel: Channel index
            unit: Unit index

        Returns:
            Array of waveforms, shape (n_samples, n_spikes), or None if no data

        Raises:
            ValueError: If channel or unit is out of range
        """
        if channel < 0 or channel >= self.number_of_channels:
            raise ValueError(f"Channel {channel} out of range")
        if unit < 0 or unit >= self.number_of_units:
            raise ValueError(f"Unit {unit} out of range")

        wf_ref = self.waveforms_ref[unit, channel]
        if not wf_ref:
            return None

        wf_data = self._file[wf_ref]
        if wf_data.size <= 2:
            return None

        return wf_data[:]

    def get_unit_statistics(self) -> Dict:
        """
        Get statistics about spike units across all channels.

        Returns:
            Dictionary with statistics for each unit
        """
        stats = {}

        for unit in range(self.number_of_units):
            total_spikes = 0
            active_channels = 0

            for channel in range(self.number_of_channels):
                spike_times = self.get_spike_times(channel, unit)
                if len(spike_times) > 0:
                    total_spikes += len(spike_times)
                    active_channels += 1

            unit_name = "unsorted" if unit == 0 else f"sorted_{unit}"
            stats[f"unit_{unit + 1}"] = {
                'name':
                unit_name,
                'total_spikes':
                total_spikes,
                'active_channels':
                active_channels,
                'avg_spikes_per_channel':
                total_spikes / active_channels if active_channels > 0 else 0
            }

        return stats

    def __str__(self) -> str:
        """Return human-readable statistics about the dataset."""
        unit_stats = self.get_unit_statistics()

        stats_str = [
            "=" * 70, "DATASET STATISTICS", "=" * 70, "",
            f"File: {self.filepath.name}",
            f"Duration: {self.duration:.2f} seconds ({self.duration / 60:.2f} minutes)",
            f"Sampling rate: {self.sampling_rate:.1f} Hz", "",
            "BEHAVIORAL DATA:", f"  Total samples: {len(self.timestamps):,}",
            f"  Cursor position: {self.cursor_positions.shape}",
            f"  Finger position: {self.finger_positions.shape}",
            f"  Target position: {self.target_positions.shape}", "",
            "TRIAL INFORMATION:",
            f"  Number of trials: {self.number_of_trials}",
            f"  Trial durations: {np.min([self.trial_ends[i] - self.trial_starts[i] for i in range(self.number_of_trials)]) / self.sampling_rate:.2f}s - "
            f"{np.max([self.trial_ends[i] - self.trial_starts[i] for i in range(self.number_of_trials)]) / self.sampling_rate:.2f}s",
            "", "NEURAL DATA:",
            f"  Recording channels: {self.number_of_channels}",
            f"  Units per channel: {self.number_of_units}",
            f"  Total features: {self.number_of_channels * self.number_of_units}",
            ""
        ]

        for unit_key, unit_info in unit_stats.items():
            stats_str.extend([
                f"  {unit_key.upper()} ({unit_info['name']}):",
                f"    Total spikes: {unit_info['total_spikes']:,}",
                f"    Active channels: {unit_info['active_channels']}/{self.number_of_channels}",
                f"    Avg spikes/channel: {unit_info['avg_spikes_per_channel']:.0f}",
                ""
            ])

        stats_str.append("=" * 70)

        return "\n".join(stats_str)

    def visualize_waveforms(self,
                            channel: int = 1,
                            output_file: str = "spike_waveforms.png"):
        """
        Visualize spike waveforms for all units on a specific channel.

        Args:
            channel: Channel to visualize (default: 1)
            output_file: Output filename for the plot

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1,
                                 self.number_of_units,
                                 figsize=(5 * self.number_of_units, 4))

        if self.number_of_units == 1:
            axes = [axes]

        unit_labels = ['UNSORTED'] + [
            f'SORTED {i}' for i in range(1, self.number_of_units)
        ]

        for unit in range(self.number_of_units):
            waveforms = self.get_waveforms(channel, unit)
            ax = axes[unit]

            if waveforms is not None and waveforms.size > 2:
                # Plot first 100 waveforms
                num_to_plot = min(100, waveforms.shape[1])
                for i in range(num_to_plot):
                    ax.plot(waveforms[:, i], 'b-', alpha=0.1, linewidth=0.5)

                # Plot mean waveform
                mean_wf = np.mean(waveforms, axis=1)
                ax.plot(mean_wf, 'r-', linewidth=2, label='Mean')

                ax.set_title(
                    f"Unit {unit + 1} ({unit_labels[unit]})\n{waveforms.shape[1]} spikes"
                )
                ax.set_xlabel("Sample")
                ax.set_ylabel("Voltage (µV)")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5,
                        0.5,
                        'No spikes',
                        ha='center',
                        va='center',
                        transform=ax.transAxes,
                        fontsize=14)
                ax.set_title(f"Unit {unit + 1} ({unit_labels[unit]})")

        channel_name = self.channel_names[channel] if channel < len(
            self.channel_names) else f"Channel {channel}"
        fig.suptitle(f"Spike Waveforms - {channel_name}", fontsize=16, y=1.02)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved waveform visualization to '{output_file}'")

        return fig

    def visualize_trial_position(self,
                                 trial_number: int,
                                 output_file: str = None):
        """
        Visualize cursor trajectory and kinematics for a specific trial.

        Args:
            trial_number: Trial index to visualize
            output_file: Output filename (default: "trial_{n}_position.png")

        Returns:
            matplotlib Figure object
        """
        trial = self.get_trial(trial_number)

        if output_file is None:
            output_file = f"trial_{trial_number}_position.png"

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Cursor trajectory
        ax = axes[0, 0]
        ax.plot(trial.cursor_positions[0, :],
                trial.cursor_positions[1, :],
                'b-',
                linewidth=2,
                label='Cursor path')
        ax.plot(trial.cursor_positions[0, 0],
                trial.cursor_positions[1, 0],
                'go',
                markersize=12,
                label='Start')
        ax.plot(trial.cursor_positions[0, -1],
                trial.cursor_positions[1, -1],
                'ro',
                markersize=12,
                label='End')
        ax.plot(trial.target_position[0],
                trial.target_position[1],
                'r*',
                markersize=20,
                label='Target')
        ax.set_xlabel('X position (mm)')
        ax.set_ylabel('Y position (mm)')
        ax.set_title(f'Trial {trial_number}: Cursor Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Plot 2: X position over time
        ax = axes[0, 1]
        ax.plot(trial.timestamps,
                trial.cursor_positions[0, :],
                'b-',
                label='Cursor X')
        ax.axhline(y=trial.target_position[0],
                   color='r',
                   linestyle='--',
                   label='Target X')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('X position (mm)')
        ax.set_title(f'Trial {trial_number}: X Position')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Y position over time
        ax = axes[1, 0]
        ax.plot(trial.timestamps,
                trial.cursor_positions[1, :],
                'b-',
                label='Cursor Y')
        ax.axhline(y=trial.target_position[1],
                   color='r',
                   linestyle='--',
                   label='Target Y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Y position (mm)')
        ax.set_title(f'Trial {trial_number}: Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Distance to target over time
        ax = axes[1, 1]
        distance = np.sqrt(
            (trial.cursor_positions[0, :] - trial.target_position[0])**2 +
            (trial.cursor_positions[1, :] - trial.target_position[1])**2)
        ax.plot(trial.timestamps, distance, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to target (mm)')
        ax.set_title(f'Trial {trial_number}: Distance to Target')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved position visualization to '{output_file}'")

        return fig

    def visualize_trial_neural(self,
                               trial_number: int,
                               max_channels: int = 20,
                               output_file: str = None):
        """
        Visualize neural activity during a specific trial.

        Args:
            trial_number: Trial index to visualize
            max_channels: Maximum number of channels to show in raster plot
            output_file: Output filename (default: "trial_{n}_neural.png")

        Returns:
            matplotlib Figure object
        """
        trial = self.get_trial(trial_number)

        if output_file is None:
            output_file = f"trial_{trial_number}_neural.png"

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Cursor trajectory
        # ax1 = fig.add_subplot(gs[0, 0])
        # ax1.plot(trial.cursor_positions[0, :],
        #          trial.cursor_positions[1, :],
        #          'b-',
        #          linewidth=2,
        #          label='Cursor path')
        # ax1.plot(trial.cursor_positions[0, 0],
        #          trial.cursor_positions[1, 0],
        #          'go',
        #          markersize=12,
        #          label='Start')
        # ax1.plot(trial.cursor_positions[0, -1],
        #          trial.cursor_positions[1, -1],
        #          'ro',
        #          markersize=12,
        #          label='End')
        # ax1.plot(trial.target_position[0],
        #          trial.target_position[1],
        #          'r*',
        #          markersize=20,
        #          label='Target')
        # ax1.set_xlabel('X position (mm)')
        # ax1.set_ylabel('Y position (mm)')
        # ax1.set_title(f'Trial {trial_number}: Cursor Trajectory')
        # ax1.legend()
        # ax1.grid(True, alpha=0.3)
        # ax1.axis('equal')

        # # Plot 2: Distance to target
        # ax2 = fig.add_subplot(gs[0, 1])
        # distance = np.sqrt(
        #     (trial.cursor_positions[0, :] - trial.target_position[0])**2 +
        #     (trial.cursor_positions[1, :] - trial.target_position[1])**2)
        # ax2.plot(trial.timestamps, distance, 'g-', linewidth=2)
        # ax2.set_xlabel('Time (s)')
        # ax2.set_ylabel('Distance to target (mm)')
        # ax2.set_title(f'Trial {trial_number}: Distance to Target')
        # ax2.grid(True, alpha=0.3)

        # Plot 3: Spike raster
        ax3 = fig.add_subplot(gs[1:3, :])

        colors = ['gray', 'red', 'blue']
        unit_labels = ['u1', 'u2', 'u3']
        neuron_id = 0
        ytick_positions = []
        ytick_labels = []

        # Sort spike times by channel and unit
        sorted_keys = sorted(trial.spike_times.keys())

        for (channel,
             unit) in sorted_keys[:max_channels * self.number_of_units]:
            if channel >= max_channels:
                break

            spike_times = trial.spike_times[(channel, unit)]

            if len(spike_times) > 0:
                # Convert to trial-relative time
                spike_times_rel = spike_times - trial.start_time

                # Plot as raster
                ax3.scatter(spike_times_rel,
                            np.ones_like(spike_times_rel) * neuron_id,
                            c=colors[unit],
                            s=5,
                            alpha=0.6)

                ytick_positions.append(neuron_id)
                ytick_labels.append(
                    f"Ch{channel}-{unit_labels[unit % len(unit_labels)]}")
                neuron_id += 1

        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Neuron', fontsize=12)
        ax3.set_title(f'Trial {trial_number}: Neural Spike Raster',
                      fontsize=14)
        ax3.set_yticks(ytick_positions[::5])  # Show every 5th label
        ax3.set_yticklabels(ytick_labels[::5])
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_xlim([0, trial.duration])

        # create legend 
        for i, label in enumerate(unit_labels):
            ax3.scatter([], [], color=colors[i], label=label)

        ax3.legend(loc='upper right')

        # Plot 4: Population firing rate
        ax4 = fig.add_subplot(gs[3, :])

        # Compute population firing rate in 50ms bins
        bin_size = 0.050
        n_bins = int(np.ceil(trial.duration / bin_size))
        bin_edges = np.linspace(0, trial.duration, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        population_rate = np.zeros(n_bins)

        for (channel, unit), spike_times in trial.spike_times.items():
            spike_times_rel = spike_times - trial.start_time
            counts, _ = np.histogram(spike_times_rel, bins=bin_edges)
            population_rate += counts

        # Convert to Hz
        population_rate = population_rate / bin_size

        ax4.plot(bin_centers, population_rate, 'k-', linewidth=2)
        ax4.fill_between(bin_centers, population_rate, alpha=0.3, color='gray')
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Population Firing Rate (Hz)', fontsize=12)
        ax4.set_title(f'Trial {trial_number}: Total Population Activity',
                      fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, trial.duration])

        plt.subplots_adjust(top=1.4)

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved neural visualization to '{output_file}'")

        return fig

    def __del__(self):
        """Close the HDF5 file when the object is destroyed."""
        if hasattr(self, '_file') and self._file:
            self._file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self, '_file') and self._file:
            self._file.close()
