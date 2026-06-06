"""
octcv.mdl_lib - Model library for OCT Computer Vision project.

Provides:
    - XVolSet: Memory-efficient wrapper for file-backed volumetric datasets
    - yX_split: Split DataFrames into labels and XVolSet
    - ModelEvaluator: Batched model evaluation (ROC, CM, classification report)
    - train_model: Standalone training function with tf.data.Dataset support
    - Visualization utilities (confusion matrix, architecture plots, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

import cv2
from PIL import Image
import base64
from io import BytesIO

from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

from keras.metrics import AUC
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping

from IPython.display import display, clear_output, Markdown, HTML
import tensorflow as tf

# Local imports
scriptDIR = os.path.dirname(os.path.abspath(__file__))
octcvDIR = os.path.dirname(scriptDIR)
projectDIR = os.path.dirname(octcvDIR)
sys.path.append(projectDIR)

from octcv.arrViz import vizInputParser
from octcv.mdl_lib.callbacks import LivePlot, EpochProgressBar

modelingDIR = os.path.join(projectDIR, 'p5_Modeling')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def describe_array(array):
    """Return a one-row DataFrame with min, mean, median, max, std, sum of an array."""
    cols = ["min", "mean", "median", "max", "std", "sum"]
    vals = [array.min(), array.mean(), np.median(array), array.max(), array.std(), array.sum()]
    return pd.DataFrame([vals], columns=cols)


def describeArrayHTML(array, title='', output='return', tblwidth=None):
    """
    Render array summary stats as a styled HTML table.

    Parameters
    ----------
    array : np.ndarray
    title : str, optional
    output : str
        'return'/'html' to return HTML string, 'disp' to display inline.
    tblwidth : int or None
        Table width percentage.

    Returns
    -------
    str or None
    """
    floatFormatter = lambda x: str(round(x, 2)) if round(x) - x != 0 else f"{int(x):,d}"
    tid = np.random.randint(0, int(1e5))

    html = describe_array(array).to_html(
        index=False, border=0, justify='center',
        float_format=floatFormatter, table_id=tid
    )

    if tblwidth is None:
        tblwidth = 100 if title else 50

    styleTag = f"""
    <style>
      .dataframe[id="{tid}"] {{ border-collapse: collapse; margin: 0; width: {tblwidth}%; }}
      .dataframe td, .dataframe th {{ text-align: center !important; padding: 8px; }}
      .dataframe th {{ color: #29313d; background-color: #e0dfff; border: 1px solid #5c6e89 !important; }}
      .dataframe td {{ color: black; background-color: white; border: 1px solid #96D4D4 !important; }}
    </style>
    """

    elements = [styleTag]
    if title:
        elements.append(f'<div style="background:#585880;color:white;padding:8px;font-weight:bold;text-align:center;border-radius:6px 6px 0 0;">{title}</div>')
    elements.append(html)

    htmlContent = '\n'.join(elements)

    if output in ('html', 'return'):
        return htmlContent
    else:
        display(HTML(htmlContent))


def numpy_to_html_img(arr, cmap_name='viridis', width='40%'):
    """Convert a 2D numpy array to an inline HTML <img> tag (base64 PNG)."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    sm = plt.get_cmap(cmap_name)
    rgba_array = sm(arr)
    rgba_uint8 = (rgba_array * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    width = width if isinstance(width, str) else str(width) + 'px'
    return f'<img src="data:image/png;base64,{img_str}" style="width:{width}; height:auto;" />'


def imgStatsCompare(images, titles=('Original', 'Noisy')):
    """Display side-by-side image comparison with stats tables (HTML in Jupyter)."""
    cdiv = lambda innerHTML: f'<div style="text-align: center;">{innerHTML}</div>'
    itagL = cdiv(numpy_to_html_img(images[0]))
    itagR = cdiv(numpy_to_html_img(images[1]))
    tabL = cdiv(describeArrayHTML(images[0], title=titles[0], output='html', tblwidth=100))
    tabR = cdiv(describeArrayHTML(images[1], title=titles[1], output='html', tblwidth=100))
    html = f"""
    <table style="width: 100%; max-width: 800px; text-align: center;">
      <tr><td>{tabL}</td><td>{tabR}</td></tr>
      <tr><td>{itagL}</td><td>{itagR}</td></tr>
    </table>
    """
    display(HTML(html))


def MinMaxScaleNDArray(array, value_range=(0, 255)):
    """Scale array to value_range using min-max normalization. Returns int array."""
    resmin, resmax = value_range
    scaled = resmin + ((array - array.min()) / (array.max() - array.min())) * (resmax - resmin)
    return scaled.astype(int)


# =============================================================================
# XVolSet - MEMORY-EFFICIENT DATASET WRAPPER
# =============================================================================

class XVolSet:
    """
    Memory-efficient wrapper for file-backed volumetric image datasets.

    Stores file paths rather than loaded arrays. Provides iteration,
    batched loading, and conversion to tf.data.Dataset for training
    without loading the entire dataset into RAM.

    Parameters
    ----------
    filepaths : pd.DataFrame or pd.Series
        DataFrame with filepath column(s). If 2 columns, first is assumed
        to be 'normalized_array' and second is 'display_volume'.
    default_load_normalized : bool
        When True, .load() uses the normalized_array column (if available).
    """

    def __init__(self, filepaths, default_load_normalized=True):
        self.filepaths = filepaths
        self.shape = filepaths.shape
        self.default_load_normalized = default_load_normalized
        self._volume_shape = None  # cached single-volume shape

    def __getitem__(self, index):
        if isinstance(index, int):
            sub_df = self.filepaths.iloc[[index]]
        else:
            sub_df = self.filepaths.iloc[index]
        return XVolSet(filepaths=sub_df, default_load_normalized=self.default_load_normalized)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"XVolSet({len(self)})"

    @property
    def volume_shape(self):
        """Shape of a single volume (without batch/channel dims). Cached after first call."""
        if self._volume_shape is None:
            path = self._get_filepath_column().iloc[0]
            sample = np.load(path) if path.endswith('.npy') else vizInputParser(path)
            self._volume_shape = sample.shape
        return self._volume_shape

    @property
    def input_shape(self):
        """Shape suitable for model input: volume_shape + channel dim."""
        return (*self.volume_shape, 1)

    def _get_filepath_column(self, normalized=None):
        """Get the appropriate filepath column based on settings."""
        if normalized is None:
            normalized = self.default_load_normalized

        if isinstance(self.filepaths, pd.Series):
            return self.filepaths

        ncols = self.filepaths.shape[1]
        if ncols == 2:
            if normalized:
                return self.filepaths.iloc[:, 0]
            else:
                return self.filepaths.iloc[:, 1]
        elif ncols == 1:
            return self.filepaths.iloc[:, 0]
        else:
            # Multi-column: look for specific column names
            if normalized and 'normalized_array' in self.filepaths.columns:
                return self.filepaths['normalized_array']
            disp_cols = [c for c in self.filepaths.columns if c.startswith('display')]
            if disp_cols:
                return self.filepaths[disp_cols[0]]
            return self.filepaths.iloc[:, 0]

    def load(self, normalized=None):
        """
        Load ALL volumes into a single numpy array. Shape: (N, *vol_shape, 1).

        WARNING: For large datasets this will consume significant RAM.
        Prefer iter_batches() or to_tf_dataset() for training.
        """
        paths = self._get_filepath_column(normalized)
        arrays = [vizInputParser(p) for p in paths.values]
        stacked = np.stack(arrays, axis=0)
        return stacked[..., np.newaxis]

    def iter_batches(self, y, batch_size=4, normalize=True):
        """
        Yield (X_batch, y_batch) tuples, loading only batch_size volumes at a time.

        Parameters
        ----------
        y : np.ndarray
            Labels array of shape (N, ...).
        batch_size : int
            Number of volumes per batch.
        normalize : bool
            If True, scale pixel values to [0, 1] by dividing by 255.

        Yields
        ------
        X_batch : np.ndarray, shape (batch_size, *vol_shape, 1), dtype float32
        y_batch : np.ndarray, shape (batch_size, ...), dtype float32
        """
        paths = self._get_filepath_column()
        n = len(paths)
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_paths = paths.iloc[i:end].values
            X_batch = np.stack([vizInputParser(p) for p in batch_paths], axis=0)
            X_batch = X_batch[..., np.newaxis].astype(np.float32)
            if normalize:
                X_batch = X_batch / 255.0
            y_batch = y[i:end].astype(np.float32)
            yield X_batch, y_batch

    def to_tf_dataset(self, y, batch_size=4, shuffle=True, normalize=True):
        """
        Convert to a tf.data.Dataset that yields (X_batch, y_batch).

        This is the preferred way to feed data into model.fit() for large datasets.

        Parameters
        ----------
        y : np.ndarray
            Labels of shape (N, ...).
        batch_size : int
            Volumes per batch.
        shuffle : bool
            If True, shuffle the dataset each epoch. Recommended for training.
        normalize : bool
            If True, scale pixel values to [0, 1] by dividing by 255.

        Returns
        -------
        tf.data.Dataset
        """
        input_shape = self.input_shape
        label_shape = y[0].shape if y.ndim > 1 else ()

        def gen():
            yield from self.iter_batches(y, batch_size, normalize=normalize)

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *label_shape), dtype=tf.float32) if label_shape else
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(self) // batch_size, 500), reshuffle_each_iteration=True)
        return ds.prefetch(tf.data.AUTOTUNE)

    def describe(self):
        """Print summary statistics (computes lazily over tf.data pipeline)."""
        paths = self._get_filepath_column()
        n = len(paths)
        vol_shape = self.volume_shape

        # Compute size on disk
        total_bytes = sum(os.path.getsize(p) for p in paths.values)
        if total_bytes > 1e9:
            size_str = f"{total_bytes / 1e9:.2f} GB"
        elif total_bytes > 1e6:
            size_str = f"{total_bytes / 1e6:.2f} MB"
        else:
            size_str = f"{total_bytes / 1e3:.2f} kB"

        full_shape = (n, *vol_shape, 1)
        n_items = int(np.prod(full_shape))

        # Sample-based stats (load a few to estimate)
        sample_indices = np.random.choice(n, size=min(50, n), replace=False)
        sample_vals = []
        for idx in sample_indices:
            arr = vizInputParser(paths.iloc[idx])
            sample_vals.extend([arr.min(), arr.max(), arr.mean(), arr.std()])

        sample_arrays = [vizInputParser(paths.iloc[idx]) for idx in sample_indices[:10]]
        all_pixels = np.concatenate([a.flatten() for a in sample_arrays])

        info = {
            'shape': str(full_shape),
            'dtype': sample_arrays[0].dtype,
            'ndim': len(full_shape),
            'n_items': f"{n_items:,}",
            'disk_size': size_str,
            'min': f"{all_pixels.min():.2f}",
            'max': f"{all_pixels.max():.2f}",
            'mean': f"{all_pixels.mean():.2f}",
            'std': f"{all_pixels.std():.2f}",
        }
        df = pd.DataFrame([info])
        display(df)


# =============================================================================
# yX_split - DATASET SPLITTING
# =============================================================================

def yX_split(df, one_hot_drop=True, default_load_normalized=True,
             display_filepath_columns=None):
    """
    Split a DataFrame into labels and XVolSet predictor set.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'dx_class' and 'glaucoma' columns, plus filepath column(s).
    one_hot_drop : bool
        If True, y is shape (N, 1) instead of (N, 2).
    default_load_normalized : bool
        Whether XVolSet defaults to loading normalized arrays.
    display_filepath_columns : str, list, or None
        Column name(s) for display volume paths. If None, auto-detects.

    Returns
    -------
    y_labels : np.ndarray of shape (N,) - string class labels
    y : np.ndarray of shape (N, 1) or (N, 2) - numeric targets
    X : XVolSet instance
    """
    y_labels = df['dx_class'].values
    yvals = df['glaucoma'].values

    if one_hot_drop:
        y = yvals.reshape(-1, 1)
    else:
        yinv = (1 - yvals)
        y = np.stack((yvals, yinv), axis=1)

    # Determine filepath columns for XVolSet
    if display_filepath_columns is None:
        disp_cols = [c for c in df.columns if c.startswith('display')]
        disp_cols = disp_cols[:1] if disp_cols else []
    elif isinstance(display_filepath_columns, str):
        disp_cols = [display_filepath_columns]
    else:
        disp_cols = list(display_filepath_columns)

    # Try to include normalized_array column
    if 'normalized_array' in df.columns:
        X_paths = df[['normalized_array'] + disp_cols]
    elif disp_cols:
        X_paths = df[disp_cols]
    else:
        raise ValueError("No filepath columns found in DataFrame.")

    X = XVolSet(X_paths, default_load_normalized=default_load_normalized)
    return y_labels, y, X


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_dataset, valid_dataset,
                monitor='val_auc', patience=3,
                restore_best_weights=True,
                learning_rate=1e-4,
                loss='binary_crossentropy',
                metrics=None,
                epochs=100,
                verbose=1,
                live_plot=False,
                metrics_to_plot=None):
    """
    Compile and train a model on tf.data.Dataset inputs.

    Parameters
    ----------
    model : keras.Model
        Uncompiled or compiled model.
    train_dataset : tf.data.Dataset
        Yields (X_batch, y_batch) for training.
    valid_dataset : tf.data.Dataset
        Yields (X_batch, y_batch) for validation.
    monitor : str
        Metric to monitor for early stopping.
    patience : int
        Epochs to wait before stopping.
    restore_best_weights : bool
        Restore model weights from the best epoch.
    learning_rate : float
        Learning rate for Nadam optimizer.
    loss : str
        Loss function name.
    metrics : list or None
        Metrics to track. Defaults to ['accuracy', AUC].
    epochs : int
        Maximum training epochs.
    verbose : int
        Keras verbosity (0=silent, 1=progress bar, 2=one line per epoch).
    live_plot : bool
        Use LivePlot callback (Jupyter notebook live chart).
    metrics_to_plot : list or None
        Which metrics to show in LivePlot. Defaults to ['auc', 'val_auc'].

    Returns
    -------
    history : keras.callbacks.History
    """
    if metrics is None:
        metrics = ['accuracy', AUC(curve='ROC', name='auc')]
    if metrics_to_plot is None:
        metrics_to_plot = ['auc', 'val_auc']

    model.compile(
        optimizer=Nadam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

    callbacks = [
        EarlyStopping(monitor=monitor, patience=patience,
                      restore_best_weights=restore_best_weights)
    ]

    if live_plot:
        callbacks.append(LivePlot(metrics_to_plot=metrics_to_plot))
        if verbose > 0:
            verbose = 0  # LivePlot replaces stdout output

    if verbose == 0 and not live_plot:
        callbacks.append(EpochProgressBar())

    start = datetime.now()
    if not live_plot and verbose > 0:
        print(f"Training: {model.name} | Started: {start:%Y-%m-%d %H:%M:%S}")
        print('━' * 50)

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    end = datetime.now()
    if not live_plot and verbose > 0:
        print('━' * 50)
        print(f"Finished: {end:%Y-%m-%d %H:%M:%S} | Duration: {end - start}")

    return history


# =============================================================================
# MODEL EVALUATOR (EVALUATION ONLY - BATCHED)
# =============================================================================

class ModelEvaluator:
    """
    Evaluate a trained model using batched prediction (memory-safe).

    Parameters
    ----------
    model : keras.Model
        Trained model.
    history : keras.callbacks.History
        Training history from model.fit().
    eval_dataset : tf.data.Dataset
        Dataset yielding (X_batch, y_batch) for evaluation.
    class_labels : list of str
        Human-readable class names. Default: ['Normal', 'Glaucoma'].
    """

    def __init__(self, model, history, eval_dataset, class_labels=None):
        self.model = model
        self.history = history
        self._eval_dataset = eval_dataset
        self.class_labels = class_labels or ['Normal', 'Glaucoma']

        # Lazy-computed
        self._y_prob = None
        self._y_true = None
        self._threshold = None

    @property
    def y_prob(self):
        """Predicted probabilities (computed lazily via batched inference)."""
        if self._y_prob is None:
            self._run_prediction()
        return self._y_prob

    @property
    def y_true(self):
        """Ground truth labels from eval dataset."""
        if self._y_true is None:
            self._run_prediction()
        return self._y_true

    @property
    def threshold(self):
        """Optimal classification threshold (Youden's J statistic)."""
        if self._threshold is None:
            self._threshold = self.optimal_threshold()
        return self._threshold

    @property
    def y_pred(self):
        """Binary predictions using optimal threshold."""
        return (self.y_prob >= self.threshold).astype(int)

    def _run_prediction(self):
        """Run batched prediction over eval dataset."""
        probs, trues = [], []
        for X_batch, y_batch in self._eval_dataset:
            probs.append(self.model.predict(X_batch, verbose=0))
            trues.append(y_batch.numpy() if hasattr(y_batch, 'numpy') else y_batch)
        self._y_prob = np.concatenate(probs, axis=0).flatten()
        self._y_true = np.concatenate(trues, axis=0).flatten()

    def optimal_threshold(self):
        """Find threshold maximizing TPR - FPR (Youden's J statistic)."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        return float(thresholds[np.argmax(tpr - fpr)])

    def roc_auc(self):
        """Compute ROC AUC score."""
        return roc_auc_score(self.y_true, self.y_prob)

    def plot_roc(self, ax=None, figsize=(6, 5)):
        """Plot ROC curve with optimal threshold marked."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        auc_val = roc_auc_score(self.y_true, self.y_prob)
        opt_idx = np.argmax(tpr - fpr)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.scatter(fpr[opt_idx], tpr[opt_idx], c='red', s=80, zorder=5,
                   label=f'Threshold = {thresholds[opt_idx]:.3f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {self.model.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_cm(self, ax=None, figsize=(5, 4)):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.class_labels)
        disp.plot(values_format='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {self.model.name}')
        return ax

    def plot_history(self, figsize=(14, 4)):
        """Plot training history (loss, accuracy, AUC)."""
        h = self.history.history
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, (metric, title) in zip(axes, [
            ('loss', 'Loss'), ('accuracy', 'Accuracy'), ('auc', 'AUC')
        ]):
            if metric in h:
                ax.plot(h[metric], label='Train')
            if f'val_{metric}' in h:
                ax.plot(h[f'val_{metric}'], label='Validation')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Training History - {self.model.name}', fontsize=13)
        plt.tight_layout()
        return fig

    def classification_report_str(self):
        """Return sklearn classification report as string."""
        return classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_labels
        )

    def summary(self, figsize=(14, 8)):
        """Display complete evaluation summary: history, ROC, CM, and report."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)

        # Training history across top row
        h = self.history.history
        for i, (metric, title) in enumerate([
            ('loss', 'Loss'), ('accuracy', 'Accuracy'), ('auc', 'AUC')
        ]):
            ax = fig.add_subplot(gs[0, i])
            if metric in h:
                ax.plot(h[metric], label='Train')
            if f'val_{metric}' in h:
                ax.plot(h[f'val_{metric}'], label='Val')
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # ROC curve bottom-left
        self.plot_roc(ax=fig.add_subplot(gs[1, 0]))

        # Confusion matrix bottom-center
        self.plot_cm(ax=fig.add_subplot(gs[1, 1]))

        # Text report bottom-right
        ax_text = fig.add_subplot(gs[1, 2])
        ax_text.axis('off')
        report = self.classification_report_str()
        ax_text.text(0.05, 0.95, report, transform=ax_text.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax_text.set_title('Classification Report', fontsize=11)

        fig.suptitle(f'Model Evaluation: {self.model.name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

        print(f"\nAUC: {self.roc_auc():.4f} | Threshold: {self.threshold:.4f}")


# =============================================================================
# LEGACY MODEL EVALUATOR (Part 1 compatibility shim)
# =============================================================================

class ModelEvaluatorLegacy:
    """
    Legacy ModelEvaluator that supports the Part 1 API.

    Accepts DataFrames for train/valid/eval sets, handles yX_split internally,
    and provides train_and_evaluate() and vkPlotArchitecture().

    This class is NOT used directly — it's instantiated automatically by
    ModelEvaluator when it detects old-style positional arguments.
    """

    def __init__(self, model, train_set=None, validation_set=None,
                 test_set=None, model_name=None):
        self.model = model
        self.model_name = model_name or model.name
        self._train_df = train_set
        self._valid_df = validation_set
        self._test_df = test_set

    def vkPlotArchitecture(self):
        """Plot model architecture using visualkeras (if available)."""
        try:
            import visualkeras
            return visualkeras.layered_view(self.model, legend=True)
        except ImportError:
            print("visualkeras not installed. Falling back to model.summary().")
            self.model.summary()

    def train_and_evaluate(self, batch_size=4, verbose=1,
                           save_performance_metrics=False, epochs=100,
                           patience=8, learning_rate=1e-4):
        """
        Train the model and evaluate on the test set (legacy workflow).

        Splits DataFrames via yX_split, creates tf.data.Datasets,
        trains with train_model(), then runs ModelEvaluator.summary().
        """
        # Split datasets
        _, y_train, X_train = yX_split(self._train_df)
        _, y_valid, X_valid = yX_split(self._valid_df)
        _, y_eval, X_eval = yX_split(self._test_df)

        # Create tf.data.Datasets
        train_ds = X_train.to_tf_dataset(y_train.flatten(), batch_size=batch_size,
                                         shuffle=True, normalize=True)
        valid_ds = X_valid.to_tf_dataset(y_valid.flatten(), batch_size=batch_size,
                                         shuffle=False, normalize=True)
        eval_ds = X_eval.to_tf_dataset(y_eval.flatten(), batch_size=batch_size,
                                       shuffle=False, normalize=True)

        # Train
        history = train_model(
            self.model, train_ds, valid_ds,
            learning_rate=learning_rate,
            patience=patience,
            epochs=epochs,
            verbose=verbose
        )

        # Evaluate
        evaluator = ModelEvaluator(self.model, history, eval_ds)
        evaluator.summary()

        self.history = history
        self.evaluator = evaluator


# Monkey-patch ModelEvaluator.__init__ to detect legacy calls
_OriginalModelEvaluator = ModelEvaluator


class ModelEvaluator(_OriginalModelEvaluator):
    """
    ModelEvaluator with backwards-compatible legacy detection.

    New API (Part 2):
        ModelEvaluator(model, history, eval_dataset)

    Legacy API (Part 1):
        ModelEvaluator(model, train_df, valid_df, eval_df)
        ModelEvaluator(model, train_set=..., validation_set=..., test_set=...)
    """

    def __new__(cls, model, *args, **kwargs):
        # Detect legacy usage: second positional arg is a DataFrame
        if args and isinstance(args[0], pd.DataFrame):
            train_df = args[0]
            valid_df = args[1] if len(args) > 1 else kwargs.get('validation_set')
            eval_df = args[2] if len(args) > 2 else kwargs.get('test_set')
            model_name = kwargs.get('model_name')
            return ModelEvaluatorLegacy(
                model, train_set=train_df,
                validation_set=valid_df, test_set=eval_df,
                model_name=model_name
            )
        # Keyword-only legacy usage
        if 'train_set' in kwargs or 'validation_set' in kwargs or 'test_set' in kwargs:
            return ModelEvaluatorLegacy(
                model,
                train_set=kwargs.get('train_set'),
                validation_set=kwargs.get('validation_set'),
                test_set=kwargs.get('test_set'),
                model_name=kwargs.get('model_name')
            )
        # New API — proceed normally
        return super().__new__(cls)


# =============================================================================
# CONFUSION MATRIX PLOTTING (STANDALONE)
# =============================================================================

def plotConfusionMatrix(y_true, y_pred, true_labels=None, ax=None,
                        title="Confusion Matrix", figsize=(5, 4), cmap='Blues'):
    """
    Plot a confusion matrix with sklearn's ConfusionMatrixDisplay.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    true_labels : list of str or None
        Display labels for classes.
    ax : matplotlib Axes or None
    title : str
    figsize : tuple
    cmap : str

    Returns
    -------
    ax : matplotlib Axes
    cm : np.ndarray - the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=true_labels)
    disp.plot(values_format='d', cmap=cmap, ax=ax)
    ax.set_title(title)
    return ax, cm
