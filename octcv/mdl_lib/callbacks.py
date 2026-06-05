"""
octcv.mdl_lib.callbacks - Custom Keras callbacks for training visualization.

Provides:
    - EpochProgressBar: Inline progress bar for silent training (verbose=0)
    - LivePlot: Real-time matplotlib training metrics plot in Jupyter
    - LiveCapture: Tabular live display of training metrics
"""

import sys
import time
import tensorflow as tf
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

def ftime(total_seconds,use_abbreviated_units=False):
    '''
    Function to convert a total number of seconds into a human-readable string with tiered units (years, months, days, hours, minutes, seconds)
    
    Parameters
    ----------
    total_seconds : float
        Total number of seconds to convert
    use_abbreviated_units : bool, optional
        If True, use abbreviated units (yr, mo, d, hr, min, s), otherwise use full units (years, months, days, hours, minutes, seconds), by default False
    '''
    start = datetime(1, 1, 1)
    end = start + timedelta(seconds=total_seconds)
    rd = relativedelta(end, start)
    values = [rd.years, rd.months, rd.days, rd.hours, rd.minutes, rd.seconds, 
            #  rd.microseconds
            ]
    
    if use_abbreviated_units:
        mu = '\N{GREEK SMALL LETTER MU}'
        units = ['yr', 'mo', 'd', 'hr', 'min', 's', 
                #  mu + 's'
                 ]
    else:
        units = ['years', 'months', 'days', 'hours', 'minutes', 'seconds',
                #   'microseconds'
                  ]
        
    result = ', '.join(f'{v} {n}' for v, n in zip(values, units) if v)
    return result

class EpochProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.target = self.params['steps']  # batches per epoch
        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        step = batch + 1
        loss = logs.get("loss", 0)
        
        progress = step / self.target
        bar_length = 30
        filled = int(bar_length * progress)

        bar = "=" * filled + "." * (bar_length - filled)

        elapsed = time.time() - self.start_time

        msg = (
            f"\rEpoch {self.current_epoch}/{self.epochs} "
            f"[{bar}] {step}/{self.target} "
            f"- loss: {loss:.4f} "
            f"- {elapsed:.1f}s"
        )

        sys.stdout.write(msg)
        sys.stdout.flush()

class LivePlot(tf.keras.callbacks.Callback):
    """
    Real-time training visualization in Jupyter notebooks.

    Shows two panels:
    - Left: Epoch-level metrics (AUC, val_AUC, etc.) updated after each epoch
    - Right: Intra-epoch batch loss/accuracy updated every ~1 second

    Parameters
    ----------
    metrics_to_plot : list of str or 'all'
        Which epoch-level metrics to plot. Default: 'all'.
    update_interval : float
        Minimum seconds between intra-epoch plot refreshes. Default: 1.0.
    """

    def __init__(self, metrics_to_plot='all', update_interval=1.0):
        super().__init__()
        self.metrics_to_plot = metrics_to_plot
        self.update_interval = update_interval

        # Epoch-level data
        self.epoch_numbers = []
        self.metric_scores = {}
        self.epoch_durations = []
        self.total_elapsed_time = 0

        # Intra-epoch (batch-level) data
        self._batch_losses = []
        self._batch_accs = []
        self._batch_steps = []
        self._last_draw_time = 0
        self._epoch_start_time = None
        self._current_epoch = 0
        self._total_steps = 0

    def on_train_begin(self, logs=None):
        self._total_steps = self.params.get('steps', 0)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()
        self._current_epoch = epoch + 1
        self._batch_losses = []
        self._batch_accs = []
        self._batch_steps = []
        self._last_draw_time = 0

    def on_batch_end(self, batch, logs=None):
        self._batch_steps.append(batch + 1)
        self._batch_losses.append(logs.get('loss', 0))
        self._batch_accs.append(logs.get('auc', logs.get('accuracy', 0)))

        now = time.time()
        if now - self._last_draw_time >= self.update_interval:
            self._last_draw_time = now
            self._draw()

    def on_epoch_end(self, epoch, logs=None):
        # Record epoch-level metrics
        for metric, score in logs.items():
            if metric not in self.metric_scores:
                self.metric_scores[metric] = []
            self.metric_scores[metric].append(score)
        self.epoch_numbers.append(int(epoch) + 1)

        # Timing
        duration = time.time() - self._epoch_start_time
        self.epoch_durations.append(duration)
        self.total_elapsed_time = sum(self.epoch_durations)

        # Final draw for this epoch
        self._draw()

    def _draw(self):
        """Render the combined plot (epoch metrics + intra-epoch progress)."""
        clear_output(wait=True)

        has_epoch_data = len(self.epoch_numbers) > 0
        fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                                 gridspec_kw={'width_ratios': [3, 2]})
        ax_epoch, ax_batch = axes

        # --- LEFT PANEL: Epoch-level metrics ---
        if has_epoch_data:
            plot_metrics = (self.metric_scores.keys() if self.metrics_to_plot == 'all'
                           else self.metrics_to_plot)
            for metric, scores in self.metric_scores.items():
                if metric in plot_metrics:
                    parts = metric.split('_')
                    label = f"{'Val ' if parts[0]=='val' else ''}{parts[-1].upper()}"
                    style = '--' if 'val' in metric else '-'
                    ax_epoch.plot(self.epoch_numbers, scores,
                                 marker='o', markersize=4, linestyle=style,
                                 alpha=0.8, label=label)
            ax_epoch.set_ylim(0, 1)
            ax_epoch.set_xlabel('Epoch')
            ax_epoch.set_ylabel('Score')
            ax_epoch.set_title('Epoch Metrics')
            ax_epoch.legend(fontsize=8, loc='lower right')
            ax_epoch.grid(True, alpha=0.3)
            ax_epoch.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax_epoch.text(0.5, 0.5, 'Waiting for first epoch...',
                         ha='center', va='center', fontsize=12, color='gray')
            ax_epoch.set_title('Epoch Metrics')

        # --- RIGHT PANEL: Intra-epoch batch progress ---
        if self._batch_steps:
            ax_batch.plot(self._batch_steps, self._batch_losses,
                         color='#e74c3c', alpha=0.7, linewidth=1, label='Loss')
            ax_batch_twin = ax_batch.twinx()
            ax_batch_twin.plot(self._batch_steps, self._batch_accs,
                              color='#2ecc71', alpha=0.7, linewidth=1, label='AUC/Acc')
            ax_batch_twin.set_ylim(0, 1)
            ax_batch_twin.set_ylabel('AUC/Acc', color='#2ecc71', fontsize=9)
            ax_batch_twin.tick_params(axis='y', labelcolor='#2ecc71')

            ax_batch.set_xlabel(f'Batch (Epoch {self._current_epoch})')
            ax_batch.set_ylabel('Loss', color='#e74c3c', fontsize=9)
            ax_batch.tick_params(axis='y', labelcolor='#e74c3c')
            ax_batch.set_title(f'Epoch {self._current_epoch} Progress')

            # Progress info
            elapsed = time.time() - self._epoch_start_time
            pct = (self._batch_steps[-1] / self._total_steps * 100) if self._total_steps else 0
            info = f"{self._batch_steps[-1]}/{self._total_steps} ({pct:.0f}%) - {elapsed:.0f}s"
            ax_batch.text(0.5, 1.02, info, transform=ax_batch.transAxes,
                         ha='center', fontsize=9, color='gray')
        else:
            ax_batch.text(0.5, 0.5, 'Starting...', ha='center', va='center',
                         fontsize=12, color='gray')
            ax_batch.set_title('Batch Progress')

        # --- Footer info ---
        ttt = ftime(self.total_elapsed_time, use_abbreviated_units=True) if self.total_elapsed_time else '0s'
        fig.suptitle(f'Total Training Time: {ttt}', fontsize=10, color='gray', y=0.02)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()

class LiveCapture(tf.keras.callbacks.Callback):
    def __init__(self, show_every='epoch'):
        super().__init__()
        self.rows = []
        self.show_every = show_every  # 'batch' or 'epoch'

    def _record(self, step, logs):
        row = {'step': step, **{k: float(v) for k, v in (logs or {}).items()}}
        self.rows.append(row)
        # live table in Jupyter
        clear_output(wait=True)
        display(pd.DataFrame(self.rows).round(5).tail(20))  # show last 20

    def on_train_batch_end(self, batch, logs=None):
        if self.show_every == 'batch':
            self._record(step=batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.show_every == 'epoch':
            self._record(step=epoch+1, logs=logs)

