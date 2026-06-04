"""
PART 2 - Modeling Workflow
==========================
Run this as cells in PPs-Modeling.ipynb (after the augmentation/split sections),
or execute directly as a script.

This replaces cells 64-70+ in the notebook with the simplified API.

Prerequisites:
    - augTRAIN, augVALID, augEVAL DataFrames exist (from earlier notebook cells)
    - OR load them fresh from the saved CSV
"""

# %% [markdown]
# ## Setup & Imports

# %%
import sys
sys.path.append('..')

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from octcv.mdl_lib import yX_split, XVolSet, ModelEvaluator, train_model
from octcv.mdl_lib.architectures import buildSequential, buildResNet, buildAttnNN

# Suppress TF info/warning logs (benign GPU registration messages)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %% [markdown]
# ## Load & Split Data

# %%
# Load augmented metadata
aug_df = pd.read_csv('../datasrc/volumeOCT-AUGMENTED_metadata.csv')
print(f"Augmented dataset: {len(aug_df)} entries")
print(f"Augmentations: {aug_df['augmentation'].unique()}")

# Load original train/test splits (from Capstone 2)
v_train = pd.read_csv('../p5_Modeling/volumes_train.csv')
v_test = pd.read_csv('../p5_Modeling/volumes_TEST.csv')

# %%
# Pivot augmented data to wide format (one row per original volume, columns per augmentation)
nonpath_columns = ['dx_class', 'glaucoma', 'PIN', 'laterality', 'left_eye', 'set']
augmentations = aug_df['augmentation'].unique()

augpiv = aug_df.pivot(
    index=nonpath_columns,
    columns='augmentation',
    values='display_volume'
).rename_axis(None, axis=1).reset_index()

# Merge with train/test splits and melt back to long format
augpiv_train = pd.merge(
    v_train, augpiv[list(augmentations)],
    left_on='display_volume', right_on='original', how='left'
).drop(columns='display_volume')

augTRAIN = pd.melt(
    augpiv_train, id_vars=nonpath_columns,
    var_name='augmentation', value_vars=augmentations,
    value_name='display_volume'
)

augpiv_test = pd.merge(
    v_test, augpiv[list(augmentations)],
    left_on='display_volume', right_on='original', how='left'
).drop(columns='display_volume')

augTEST = pd.melt(
    augpiv_test, id_vars=nonpath_columns,
    var_name='augmentation', value_vars=augmentations,
    value_name='display_volume'
)

augVALID, augEVAL = train_test_split(augTEST, test_size=0.5, random_state=42)

print(f"Train: {len(augTRAIN)} | Valid: {len(augVALID)} | Eval: {len(augEVAL)}")

# %% [markdown]
# ## Create tf.data.Dataset Pipelines

# %%
# Split into y and X (XVolSet wraps file paths, never loads all into RAM)
_, ay_train, aX_train = yX_split(augTRAIN, display_filepath_columns='display_volume')
_, ay_valid, aX_valid = yX_split(augVALID, display_filepath_columns='display_volume')
_, ay_eval, aX_eval = yX_split(augEVAL, display_filepath_columns='display_volume')

BATCH_SIZE = 4

# Convert to tf.data.Dataset (loads volumes lazily in batches)
train_ds = aX_train.to_tf_dataset(ay_train, batch_size=BATCH_SIZE)
valid_ds = aX_valid.to_tf_dataset(ay_valid, batch_size=BATCH_SIZE)
eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)

print(f"Input shape: {aX_train.input_shape}")
print(f"Batches per epoch (train): ~{len(aX_train) // BATCH_SIZE}")

# %% [markdown]
# ## Train Model 1: Sequential CNN

# %%
model_seq = buildSequential()
model_seq.summary()

# %%
history_seq = train_model(
    model_seq, train_ds, valid_ds,
    epochs=50,
    patience=5,
    learning_rate=1e-4,
    live_plot=True,  # Set False if not in Jupyter
    metrics_to_plot=['auc', 'val_auc']
)

# %% [markdown]
# ## Evaluate Sequential Model

# %%
# Re-create eval dataset (generators are single-use)
eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)

evaluator_seq = ModelEvaluator(model_seq, history_seq, eval_ds)
evaluator_seq.summary()

# %% [markdown]
# ## Train Model 2: ResNet-Like

# %%
model_res = buildResNet()

# Re-create datasets (generators exhausted after one pass)
train_ds = aX_train.to_tf_dataset(ay_train, batch_size=BATCH_SIZE)
valid_ds = aX_valid.to_tf_dataset(ay_valid, batch_size=BATCH_SIZE)

history_res = train_model(
    model_res, train_ds, valid_ds,
    epochs=50,
    patience=5,
    learning_rate=1e-4,
    live_plot=True
)

# %%
eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)
evaluator_res = ModelEvaluator(model_res, history_res, eval_ds)
evaluator_res.summary()

# %% [markdown]
# ## Train Model 3: Attention Network

# %%
model_attn = buildAttnNN()

train_ds = aX_train.to_tf_dataset(ay_train, batch_size=BATCH_SIZE)
valid_ds = aX_valid.to_tf_dataset(ay_valid, batch_size=BATCH_SIZE)

history_attn = train_model(
    model_attn, train_ds, valid_ds,
    epochs=50,
    patience=5,
    learning_rate=1e-4,
    live_plot=True
)

# %%
eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)
evaluator_attn = ModelEvaluator(model_attn, history_attn, eval_ds)
evaluator_attn.summary()

# %% [markdown]
# ## Model Comparison

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
models_data = [
    (evaluator_seq, 'Sequential'),
    (evaluator_res, 'ResNet-Like'),
    (evaluator_attn, 'Attention'),
]

for ax, (ev, name) in zip(axes, models_data):
    ev.plot_roc(ax=ax)
    ax.set_title(f'{name}\nAUC = {ev.roc_auc():.4f}')

plt.suptitle('ROC Comparison - Augmented Dataset', fontsize=14)
plt.tight_layout()
plt.show()

# Print comparison table
print(f"\n{'Model':<15} {'AUC':<8} {'Threshold':<12}")
print('-' * 35)
for ev, name in models_data:
    print(f"{name:<15} {ev.roc_auc():.4f}   {ev.threshold:.4f}")

# %% [markdown]
# ## Hyperparameter Tuning (Sequential Model)
#
# Fine-tune learning rate and patience using the best-performing architecture.

# %%
# Grid of learning rates to try
learning_rates = [5e-5, 1e-4, 3e-4, 5e-4]
results = []

for lr in learning_rates:
    print(f"\n{'='*40}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*40}")

    model = buildSequential(model_name=f'seq_lr{lr}')

    train_ds = aX_train.to_tf_dataset(ay_train, batch_size=BATCH_SIZE)
    valid_ds = aX_valid.to_tf_dataset(ay_valid, batch_size=BATCH_SIZE)

    history = train_model(
        model, train_ds, valid_ds,
        epochs=30,
        patience=4,
        learning_rate=lr,
        verbose=2  # One line per epoch
    )

    # Quick eval
    eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)
    ev = ModelEvaluator(model, history, eval_ds)
    auc = ev.roc_auc()
    results.append({'lr': lr, 'auc': auc, 'epochs': len(history.history['loss'])})
    print(f"  → AUC: {auc:.4f} in {results[-1]['epochs']} epochs")

# %%
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
print("\nHyperparameter Tuning Results:")
print(results_df.to_string(index=False))

best_lr = results_df.iloc[0]['lr']
print(f"\nBest learning rate: {best_lr}")

# %% [markdown]
# ## Final Training with Best Hyperparameters

# %%
final_model = buildSequential(model_name='final_augmented')

train_ds = aX_train.to_tf_dataset(ay_train, batch_size=BATCH_SIZE)
valid_ds = aX_valid.to_tf_dataset(ay_valid, batch_size=BATCH_SIZE)

final_history = train_model(
    final_model, train_ds, valid_ds,
    epochs=80,
    patience=8,
    learning_rate=best_lr,
    live_plot=True
)

# %%
eval_ds = aX_eval.to_tf_dataset(ay_eval, batch_size=BATCH_SIZE)
final_evaluator = ModelEvaluator(final_model, final_history, eval_ds)
final_evaluator.summary()

# %% [markdown]
# ## Save Final Model

# %%
save_dir = os.path.join('..', 'p5_Modeling', 'models', 'augmented_final')
os.makedirs(save_dir, exist_ok=True)
final_model.save(os.path.join(save_dir, 'augmented_final.keras'))
print(f"Model saved to: {save_dir}")
