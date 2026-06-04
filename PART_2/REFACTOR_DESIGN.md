# ModelEvaluator & mdl_lib Refactor Design

## Problem Summary

The `ModelEvaluator` class and supporting utilities (`XVolSet`, `yX_split`) were designed assuming all data fits in RAM. With the augmented dataset (5328 train / 666 validation / 666 eval volumes at 64×128×64 each), loading everything via `.load()` is no longer feasible (~2.8 GB for train alone as float32).

The workaround (cells 69-70 in PPs-Modeling.ipynb) already works by using `tf.data.Dataset` from generators, but `ModelEvaluator` can't consume these directly.

## Current Architecture (What Exists)

```
ModelEvaluator(model, train_df, valid_df, test_df)
    ├── Constructor calls yX_split() on each df → creates XVolSet instances
    ├── .train() → internally creates generators → tf.data.Dataset → model.fit()
    ├── .predict() → self.X_test.load() (LOADS ALL INTO RAM)
    ├── .optimalThreshold() → needs y_prob (from predict)
    ├── .plotCM() → needs y_pred
    ├── .plotROC() → needs y_prob, y_true
    └── .classificationReport() → needs y_pred, y_true
```

**Key issues:**
1. Constructor does too much (yX_split on every set immediately)
2. `.predict()` calls `X_test.load()` — RAM killer
3. `.train()` re-creates the same generator/tf.data logic that already exists standalone
4. The class tries to be both a trainer AND evaluator — separate concerns

## Proposed Simplified Architecture

### Principle: Separate training from evaluation, accept tf.data.Dataset natively.

```python
# NEW: Lightweight training function (replaces ModelEvaluator.train)
def train_model(model, train_dataset, valid_dataset, 
                monitor='val_auc', patience=3, epochs=100,
                learning_rate=1e-4, live_plot=False):
    """Train a compiled model on tf.data.Dataset inputs. Returns history."""
    ...

# NEW: Simplified ModelEvaluator (evaluation only, works in batches)
class ModelEvaluator:
    def __init__(self, model, history, eval_dataset, class_labels=None):
        """
        model: trained keras model
        history: keras History object from model.fit()
        eval_dataset: tf.data.Dataset yielding (X_batch, y_batch)
        class_labels: list of str, e.g. ['Normal', 'Glaucoma']
        """
        self.model = model
        self.history = history
        self.class_labels = class_labels or ['Negative', 'Positive']
        
        # Predict in batches on init (or lazily)
        self._y_prob = None
        self._y_true = None
        self._eval_dataset = eval_dataset
    
    @property
    def y_prob(self):
        if self._y_prob is None:
            self._predict_batched()
        return self._y_prob
    
    @property
    def y_true(self):
        if self._y_true is None:
            self._predict_batched()
        return self._y_true
    
    def _predict_batched(self):
        """Predict on eval dataset in batches without loading all into RAM."""
        probs, trues = [], []
        for X_batch, y_batch in self._eval_dataset:
            probs.append(self.model.predict(X_batch, verbose=0))
            trues.append(y_batch.numpy())
        self._y_prob = np.concatenate(probs, axis=0)
        self._y_true = np.concatenate(trues, axis=0)
    
    def optimal_threshold(self, method='roc_curve'): ...
    def predict(self, threshold=None): ...  # returns y_pred
    def plot_roc(self, ax=None): ...
    def plot_cm(self, ax=None, **kwargs): ...
    def plot_history(self, metrics=None, ax=None): ...
    def classification_report(self): ...
    def summary_figure(self): ...  # combines all plots in one figure
```

### XVolSet Simplification

Keep `XVolSet` but remove the summary stats computation from init (it's expensive and rarely needed at construction time). The key method is `.iter_batches()` — that's all training needs.

```python
class XVolSet:
    def __init__(self, filepaths):
        self.filepaths = filepaths  # DataFrame or Series of .npy paths
    
    def __len__(self): return len(self.filepaths)
    def __getitem__(self, idx): ...  # returns single loaded array
    
    def iter_batches(self, y, batch_size=4):
        """Yields (X_batch, y_batch) tuples. Core method for training."""
        ...
    
    def to_tf_dataset(self, y, batch_size=4):
        """Convenience: returns a tf.data.Dataset directly."""
        def gen():
            yield from self.iter_batches(y, batch_size)
        input_shape = self[0].shape  # sample one to get shape
        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
    
    def describe(self): ...  # lazy, computed on demand
```

### Simplified yX_split

```python
def yX_split(df, filepath_col='display_volume'):
    """
    Returns (y_labels, y, X) where X is an XVolSet.
    Simplified: single filepath column, no normalized_array dual-column logic.
    """
    y_labels = df['dx_class'].values
    y = df['glaucoma'].values.reshape(-1, 1).astype(np.float32)
    X = XVolSet(df[[filepath_col]])
    return y_labels, y, X
```

## Migration Path

1. **Now (Option A):** Use direct evaluation code in notebook. No module changes needed.
2. **Next:** Refactor `ModelEvaluator` to the new design. Keep old class as `ModelEvaluatorLegacy` temporarily.
3. **Then:** Update `XVolSet` with `.to_tf_dataset()` method.
4. **Finally:** Simplify `yX_split` to the one-column version (Part 2 doesn't use normalized arrays anyway).

## What to Keep vs Remove

| Keep (Essential) | Simplify | Remove/Deprecate |
|---|---|---|
| `XVolSet.iter_batches()` | `ModelEvaluator` (eval only) | `ModelEvaluator.train()` |
| `yX_split()` | `XVolSet.describe()` (lazy) | `XVolSet.load()` for full dataset |
| `buildSequential/ResNet/Attn` | `plotConfusionMatrix()` | `get_saved_epoch()` |
| `LivePlot` callback | | `XVolSet.tfBatchedDataset()` (redundant with to_tf_dataset) |
| `EpochProgressBar` callback | | All the HTML display utilities (nice-to-have, not needed for training) |

## Notebook Workflow After Refactor

```python
# 1. Load & split data
aug_df = pd.read_csv('../datasrc/volumeOCT-AUGMENTED_metadata.csv')
# ... train_test_split logic ...
_, y_train, X_train = yX_split(augTRAIN)
_, y_valid, X_valid = yX_split(augVALID)
_, y_eval, X_eval = yX_split(augEVAL)

# 2. Create datasets
train_ds = X_train.to_tf_dataset(y_train, batch_size=4)
valid_ds = X_valid.to_tf_dataset(y_valid, batch_size=4)
eval_ds = X_eval.to_tf_dataset(y_eval, batch_size=4)

# 3. Build & train
model = buildSequential()
history = train_model(model, train_ds, valid_ds, live_plot=True)

# 4. Evaluate
evaluator = ModelEvaluator(model, history, eval_ds, 
                           class_labels=['Normal', 'Glaucoma'])
evaluator.summary_figure()
print(evaluator.classification_report())
```

Clean, readable, memory-safe.
