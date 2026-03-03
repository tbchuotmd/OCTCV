
import sys
import time
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

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
    
    def __init__(self,metrics_to_plot='all'):
        self.epoch_numbers = []
        self.metric_scores = {}
        self.metrics_to_plot = metrics_to_plot
        self.epoch_durations = []
        self.total_elapsed_time = 0
    def plot_epoch_data(self,
                        epoch_numbers:list,
                        metric_scores:dict,
                        epoch_durations:list):
        
        fig, ax = plt.subplots(figsize=(8,3.5))
        
        if self.metrics_to_plot == 'all':
            metrics_to_plot = metric_scores.keys()
        else:
            metrics_to_plot = self.metrics_to_plot

        for metric,scores in metric_scores.items():
            if metric in metrics_to_plot:
                metric_split = metric.split('_')
                if len(metric_split) > 1:
                    last = metric_split[-1]
                    first = metric_split[0]
                    metricf = f'{first.title()} {last.upper()}'
                else:
                    metricf = metric.upper()

                ax.plot(epoch_numbers, scores, 
                        marker='o', markersize=5,
                        alpha=0.6, 
                        label=metricf)
        
        plt.grid(True)
        ax.set_ylim(0,1)
        ax.set_yticks(np.linspace(0,1,9))
        # ax.set_xticks( np.arange(1,self.params['epochs'] + 1) )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Epoch', fontsize=13)
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title('Training Metrics', fontsize=15)
        handles, labels = ax.get_legend_handles_labels()

        if labels:
            ax.legend(
                bbox_to_anchor=(1.05, 1), loc='upper left', 
                borderaxespad=0.5
            )
            
            if epoch_durations:
                ept = ftime(epoch_durations[-1],use_abbreviated_units=True)
            else:
                ept = 'N/A'
            ax.text(
                1.05,0.3,
                s = f"Epoch #{epoch_numbers[-1]}\nDuration: {ept}\nTraining AUC:    {metric_scores['auc'][-1]:.4f}\nValidation AUC: {metric_scores['val_auc'][-1]:.4f}",
                transform=ax.transAxes, 
                ha='left', va='center',
                bbox=dict(facecolor='black', 
                              alpha=0.9, 
                              edgecolor='black', 
                              lw=0.5,
                              boxstyle='round,pad=0.3'),
                fontdict={'color': 'white', 
                              'weight': 'bold', 
                              'size': 10}
            )

            ttt = ftime(self.total_elapsed_time,use_abbreviated_units=True)
            ax.text(
                1.05,0.6,
                s = f"Total Training Time\n{ttt}",
                transform=ax.transAxes, 
                ha='left', va='center',
                bbox=dict(facecolor='black', 
                              alpha=0.9, 
                              edgecolor='black', 
                              lw=0.5,
                              boxstyle='round,pad=0.3'),
                fontdict={'color': 'white', 
                              'weight': 'bold', 
                              'size': 10}
            )
        plt.show()
        

    # def on_train_begin(self,logs=None):
        
    def on_epoch_begin(self, epoch, logs = None):
        self.start_time = datetime.now()
        if epoch == 0:
            self.plot_epoch_data(self.epoch_numbers, 
                                self.metric_scores,
                                self.epoch_durations)

    def on_epoch_end(self,epoch,logs=None):
        for metric,score in logs.items():
            if metric not in self.metric_scores.keys():
                self.metric_scores[metric] = []
            self.metric_scores[metric].append(score)
        self.epoch_numbers.append(int(epoch)+1)

        # update training time
        self.end_time = datetime.now()
        self.epoch_durations.append((self.end_time - self.start_time).total_seconds())
        self.total_elapsed_time = sum(self.epoch_durations)
        
        clear_output(wait=True)
        self.plot_epoch_data(self.epoch_numbers, 
                             self.metric_scores,
                             self.epoch_durations)

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

