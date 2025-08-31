# import os,sys
# sys.path.append(os.path.abspath('..'))
from octcv.arrViz import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv3D,Conv2D,Dense,MaxPool2D,MaxPool3D,Flatten,Dropout,\
BatchNormalization,GlobalAveragePooling3D,GlobalAveragePooling2D,ReLU,Input,Add
from keras.metrics import AUC
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras import Model as kerasModel

# Define class that stores arrays of filepaths, with a method to load the files as numpy arrays and format as needed
class XVolSet:
    '''
    Class for storing image paths as X (predictor variables), while easily loading all images as arrays when needed via .load() method.
    '''
    def __init__(self,filepaths):
        self.filepaths = filepaths
        self.shape = filepaths.shape
    def load(self,normalized=True):
        image_paths = self.filepaths
        # shape: (N, 64, 128, 64)
        nrows,ncols = np.array(image_paths).shape
        if ncols == 2:
            if type(image_paths)==pd.DataFrame:
                col_name = 'normalized_array' if normalized else [ c for c in image_paths.columns if c.startswith('display') ][0]
                # col_name = 'normalized_array'
                xarrs = image_paths[col_name].apply(vizInputParser).values
                xarrs = np.stack(xarrs,axis=0)
            elif type(image_paths)==np.ndarray or type(image_paths)==list:
                col_index = 0 if normalized else 1
                xarrs = np.array([vizInputParser(path) for path in image_paths[:,col_index]]) # shape: (N, 64, 128, 64)
        elif ncols == 1:
            xarrs = np.array([vizInputParser(path) for path in image_path])
        xarrs = xarrs[...,np.newaxis] # add channel dim --> (N, 64, 128, 64, 1)
        return xarrs

# Function to split datasets into y,X; but X is the XVolSet instance
def yX_split(df):
    '''
    Splits datasets into y and X, along with y_labels (strings) for downstream interpretation.

    PARAMS:
    -------
    df    :    DataFrame, formated per OCTCV project pre-processing

    RETURNS:
    --------
    (y_labels, y, X)    :     tuple

        y_labels        :     numpy.ndarray of shape (N,) with class labels in human-readable / string format for interpretation of results after fitting/training
        y               :     numpu.ndarry of shape (N,2) with hot-one encoded numeric class labels
                                    column 0 --> "glaucoma" (1 if True, 0 if False)
                                    column 1 --> "normal" (1 if True, 0 if False)
        X               :     XVolSet() instance -- contains both normalized & display image filepaths (column 0 and 1, respectively)
                                    actual image data accessible via .load() method
    '''
    # Subset class labels for ease of interpretation later on
    y_labels = df.dx_class.values
    
    # Subset numeric class target variable (`glaucoma` column of df)
    yvals = df.glaucoma.values
    # Recreate `normal` column that was dropped during one-hot encoding
    yinv =  (df.glaucoma - 1).abs().values # `yinv` for "y-inverted"
    # Combine columns into one  for y
    y = np.stack((yvals,yinv),axis=1) # shape: (N,2) where N = len(df)

    # Get both the normalized image paths and the display images paths
    disp_col = [ c for c in df.columns if c.startswith('display') ][0] # can be useful for visualization later on
    X_paths = df[['normalized_array',disp_col]] # "normalized_array" part for training / inference
    X = XVolSet(X_paths)
    return y_labels, y, X



# Helper function that creates a single residual block.
def residual_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    shortcut = x
        
    # First convolution
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second convolution
    x = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Handle potential shape mismatch for the skip connection
    # This occurs when strides > 1
    if strides != 1:
        shortcut = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add the skip connection to the main path
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x


# This code uses the functional API to construct the full model by chaining the residual blocks together.
def build_resnet_like_model(input_shape=(64, 128, 64, 1), num_classes=2):
    inputs = Input(shape=input_shape)
    
    # Initial layer: reduced filters and larger kernel
    x = Conv3D(16, kernel_size=5, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual Block 1: No change in spatial dimensions
    x = residual_block(x, filters=16, strides=1)
    
    # Residual Block 2: Reduce spatial dimensions with stride 2
    x = residual_block(x, filters=32, strides=2)
    
    # Residual Block 3: No change in spatial dimensions
    x = residual_block(x, filters=32, strides=1)
    
    # Global Average Pooling
    x = GlobalAveragePooling3D()(x)
    
    # Dense layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and return the model
    model = kerasModel(inputs, outputs)
    return model

#--------------------------------------------------------------


### EDIT TO ALLOW COMPARISONS OF PLOTS BETWEEN MODELS
### Currently, issue is: loss,acc,auc are required for the axvline in the plot
###     these all come from model.evaluate(), which requires the whole model.
###     instead, might want to save these as a separate tuple outside of `plotModelHistory` context...

# Get the best/saved epoch metrics
def get_saved_epoch(model,model_history, test_set):
    _, y_test, X_test = yX_split(test_set)
    loss, acc, auc = model.evaluate( X_test.load(), y_test )
    saved_epoch = [ i for i,v in enumerate(model_history.history['val_auc']) if round(v,4) == round(auc,4) ][0]
    return loss,acc,auc,saved_epoch

def plotModelHistory(model,model_history, test_set, ax=None, figsize=(8,4)):
    loss,acc,auc,saved_epoch = get_saved_epoch(model, model_history,test_set)

    if not ax:
        fig,ax = plt.subplots(1,1, figsize=figsize)
    
    ax.plot(model_history.history['val_auc'], label='val AUC', marker='.')
    ax.plot(model_history.history['auc'], label='train AUC', marker='.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.axvline(saved_epoch, linestyle='--', c='r', label='Saved Epoch')
    ax.text(saved_epoch+.2,.8,f"val:    {auc:.4f}\ntrain:  {model_history.history['auc'][saved_epoch]:.4f}")
    ax.legend()
    _ = ax.set_title('AUC over Epochs')