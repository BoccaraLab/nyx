"""Hypnogram, spectrogram and multi-panel figures."""

from datetime import datetime, timedelta

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from nyx.stages import COLORS as colors_dict

regions_colors = {'PAR': '#C62E65', 'PFC': '#F46036', 'CA1': '#08B2E3'}


def _setup_xaxis(ax, max_time, plot_from_time, xaxis_format):
    """Helper function to set up the x-axis."""
    use_time_of_day = isinstance(plot_from_time, datetime)
    
    if use_time_of_day:
        num_ticks = 10
        tick_positions = np.linspace(0, max_time, num_ticks)
        tick_labels = [(plot_from_time + timedelta(seconds=pos)).strftime("%H:%M")
                       for pos in tick_positions]
        ax.set_xlabel("Time of Day (HH:MM)")
    else:
        if xaxis_format == "Minutes":
            tick_positions = np.arange(0, max_time, 600)
            tick_labels = [f"{int(pos / 60)}" for pos in tick_positions]
            ax.set_xlabel("Time (min)")
        elif xaxis_format == "Hours":
            tick_positions = np.arange(0, max_time, 3600)
            tick_labels = [f"{int(pos / 3600)}" for pos in tick_positions]
            ax.set_xlabel("Time (h)")
        else:
            num_ticks = 10
            tick_positions = np.linspace(0, max_time, num_ticks)
            tick_labels = [f"{int(pos)}" for pos in tick_positions]
            ax.set_xlabel("Time (s)")

    ax.set_xlim(0, max_time)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def plot_hypnogram(epoch_data, possible_labels, title, savepath=None, ax=None, plot_from_time=0., xaxis_format="Seconds", linear=False):
    """Generalized hypnogram plotting function."""
    if not all(col in epoch_data.columns for col in ['time', 'duration', 'label']):
        raise ValueError("DataFrame must contain 'time', 'duration', and 'label' columns.")

    time = epoch_data['time'].values
    duration = epoch_data['duration'].values
    labels = epoch_data['label'].values

    present_labels = np.unique(labels)
    filtered_possible_labels = [label for label in possible_labels if label in present_labels]
    reversed_labels = list(reversed(filtered_possible_labels))
    label_to_int = {label: idx for idx, label in enumerate(reversed_labels)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 5))

    for start_time, dur, label in zip(time, duration, labels):
        if label in label_to_int:
            label_position = 0 if linear else label_to_int[label]
            current_color = colors_dict.get(label, "#000000")
            ax.add_patch(patches.Rectangle(
                (start_time, label_position),
                dur, 1,
                edgecolor=current_color, facecolor=current_color
            ))

    max_time = time[-1] + duration[-1]
    _setup_xaxis(ax, max_time, plot_from_time, xaxis_format)

    if not linear:
        ax.set_ylim(0, len(reversed_labels))
        ax.set_yticks([i + 0.5 for i in range(len(reversed_labels))])
        ax.set_yticklabels(reversed_labels)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([])

    ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(False)

def generate_custom_plot(config):
    """
    Generate a customizable plot based on the provided configuration dictionary.

    Parameters:
        config (dict): A dictionary containing the plot configuration. Example structure:
            {
                "subplots": [
                    {
                        "type": "trace",
                        "data": [x, y],
                        "color": "blue",
                        "height": 1,
                        "lw": 0.5,
                        "label": "EMG Trace"
                    },
                    {
                        "type": "spectrogram",
                        "data": [frequencies, times, spectrogram],
                        "im_range": [0, 1],
                        "palette": "jet",
                        "height": 4,
                        "label": "Spectrogram"
                    },
                    {
                        "type": "trace",
                        "data": [x, pc1],
                        "color": "red",
                        "lw": 2,
                        "threshold": 0.5,
                        "height": 1,
                        "label": "PC1 Trace"
                    },
                    {
                        "type": "hypnogram",
                        "data": epoch_data,  # DataFrame with 'time', 'duration', 'label' columns
                        "possible_labels": ["Wake", "NREM", "REM"],
                        "title": "Hypnogram",
                        "height": 2,
                        "plot_from_time": 0.,
                        "xaxis_format": "Seconds",
                        "linear": False
                    }
                ]
            }
    """

    # Calculate total height of the figure
    fig = plt.figure(figsize=config["figsize"] if "figsize" in config else (15, 10))
    gs = GridSpec(len(config["subplots"]), 2 if any(subplot["type"] == "spectrogram" for subplot in config["subplots"]) else 1, 
                  height_ratios=[subplot["height"] for subplot in config["subplots"]],
                  width_ratios=[30, 1] if any(subplot["type"] == "spectrogram" for subplot in config["subplots"]) else [1],
                  wspace=0.05)

    axes = []
    for i, subplot in enumerate(config["subplots"]):
        if subplot["type"] == "spectrogram":
            ax_main = fig.add_subplot(gs[i, 0])
            ax_colorbar = fig.add_subplot(gs[i, 1])
            axes.append((ax_main, ax_colorbar))
        else:
            ax = fig.add_subplot(gs[i, 0])
            axes.append(ax)

    for i, subplot in enumerate(config["subplots"]):
        if subplot["type"] == "trace":
            ax = axes[i]
            x, y = subplot["data"]
            ax.plot(x, y, color=subplot.get("color", "black"), lw=subplot.get("lw", 1), label=subplot.get("label", None))
            if "threshold" in subplot:
                ax.axhline(subplot["threshold"], color="red", lw=2, linestyle="--", label="Threshold")
            ax.set_ylabel(subplot.get("label", ""))
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(subplot.get("ylim", (None, None)))
            for spine in ax.spines.values():
                spine.set_visible(False)

        elif subplot["type"] == "spectrogram":
            ax, ax_colorbar = axes[i]
            frequencies, times, spectrogram = subplot["data"]
            im = ax.pcolormesh(times, frequencies, spectrogram, shading="nearest", cmap=subplot.get("palette", "jet"), rasterized=True)
            im.set_clim(*subplot.get("im_range", [None, None]))
            fig.colorbar(im, cax=ax_colorbar, orientation="vertical", label=subplot.get("label", ""))
            ax.set_ylabel("Frequency (Hz)")

        elif subplot["type"] == "hypnogram":
            ax = axes[i]
            plot_hypnogram(
                epoch_data=subplot["data"],
                possible_labels=subplot["possible_labels"],
                title=subplot.get("title", "Hypnogram"),
                ax=ax,
                plot_from_time=subplot.get("plot_from_time", 0.),
                xaxis_format=subplot.get("xaxis_format", "Seconds"),
                linear=subplot.get("linear", False)
            )

    ax.set_xlabel("Time (s)")

def plot_reconstructed_spectrogram(reconstructed_spectrogram, frequencies, times, time_window=None, title="Reconstructed Spectrogram", savepath=None, im_range=[0, 1]):
    """
    Plots the reconstructed spectrogram using generate_custom_plot, with an option to focus on a specific time window.

    Parameters:
        reconstructed_spectrogram (np.ndarray): The reconstructed spectrogram data.
        frequencies (np.ndarray): The frequencies for the y-axis.
        times (np.ndarray): The times for the x-axis.
        time_window (tuple, optional): A tuple (start_time, end_time) to slice the data. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Reconstructed Spectrogram".
        savepath (str, optional): Path to save the figure. If None, the plot is shown. Defaults to None.
        im_range (list, optional): The color limits for the spectrogram. Defaults to [0, 1].
    """
    times_to_plot = times
    spectrogram_to_plot = reconstructed_spectrogram

    if time_window:
        start_time, end_time = time_window
        time_indices = np.where((times >= start_time) & (times <= end_time))[0]
        if len(time_indices) > 0:
            times_to_plot = times[time_indices]
            # Ensure spectrogram_to_plot is correctly sliced
            spectrogram_to_plot = reconstructed_spectrogram[:, time_indices]
        else:
            print("Warning: Time window is empty or out of bounds. Plotting the full spectrogram.")

    config = {
        "figsize": (20, 5),
        "subplots": [
            {
                "type": "spectrogram",
                "data": [frequencies, times_to_plot, spectrogram_to_plot],
                "im_range": im_range,
                "palette": "jet",
                "height": 4,
                "label": "Power"
            }
        ]
    }

    generate_custom_plot(config)
    plt.suptitle(title)

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()