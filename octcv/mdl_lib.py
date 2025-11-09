# import os,sys
# sys.path.append(os.path.abspath('..'))
from octcv.arrViz import *
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from num2words import num2words

import visualkeras as vk
from PIL import ImageFont
import cv2
from datetime import datetime,timedelta

from keras.models import Sequential
from keras.layers import Conv3D,Conv2D,Dense,MaxPool2D,MaxPool3D,Flatten,Dropout,\
BatchNormalization,GlobalAveragePooling3D,GlobalAveragePooling2D,ReLU,Input,Add
from keras.metrics import AUC
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras import Model as kerasModel
from keras.utils import plot_model
from sklearn.metrics import roc_curve, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.callbacks import Callback

from IPython.display import display,clear_output
import tensorflow as tf

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
def yX_split(df,one_hot_drop=True,summary_report=False):
    '''
    Splits datasets into y and X, along with y_labels (strings) for downstream interpretation.

    PARAMS:
    -------
    df    :    DataFrame, formated per OCTCV project pre-processing
    one_hot_drop: bool, if True, will drop extra column in y
    summary_report : bool, if True, will print a summary of return value stats

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
    if one_hot_drop:
        y = y[:,0].reshape(-1,1)        

    # Get both the normalized image paths and the display images paths
    disp_col = [ c for c in df.columns if c.startswith('display') ][0] # can be useful for visualization later on
    X_paths = df[['normalized_array',disp_col]] # "normalized_array" part for training / inference
    X = XVolSet(X_paths)

    if summary_report:
        print(f"y_labels shape: {y_labels.shape}")
        print(f"y shape: {y.shape}")
        print(f"X shape {X.shape}")
        
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



def plotAUC(model_history, test_set, saved_epoch_data, ax=None, figsize=(8,4)):
    saved_loss,saved_acc,saved_auc,saved_epoch = saved_epoch_data
    
    if not ax:
        fig,ax = plt.subplots(1,1, figsize=figsize)
    
    ax.plot(model_history.history['val_auc'], label='val AUC', marker='.')
    ax.plot(model_history.history['auc'], label='train AUC', marker='.')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.axvline(saved_epoch, linestyle='--', c='r', label='Saved Epoch')
    ax.text(saved_epoch+.2,.8,f"val:    {saved_auc:.4f}\ntrain:  {model_history.history['auc'][saved_epoch]:.4f}")
    ax.legend()
    _ = ax.set_title('AUC over Epochs')


class ModelVisualizer:
    def __init__(self, model):
        self.model = model
    def text_callable(self, layer_index, layer):
        # Every other piece of text is drawn above the layer, the first one below
        above = bool(layer_index%2)

        # Get the output shape of the layer
        if hasattr(layer, 'output_shape'):
            # ───────────────────────────────
            # Legacy Keras (standalone 2.x) or TF-Keras <2.11
            # ───────────────────────────────
            # this will work so long as `layer.output_shape` exists
            output_shape = [x for x in list(layer.output.shape) if x is not None]
        else:
            # ───────────────────────────────
            # Modern TF-Keras (>= 2.11) – attribute removed
            # ───────────────────────────────
            output_shape = list(layer.output.shape)

        # If the output shape is a list of tuples, we only take the first one
        if isinstance(output_shape[0], tuple):
            output_shape = list(output_shape[0])
            output_shape = [x for x in output_shape if x is not None]

        # Variable to store text which will be drawn    
        output_shape_txt = ""

        # Create a string representation of the output shape
        for ii in range(len(output_shape)):
            output_shape_txt += str(output_shape[ii])
            if ii < len(output_shape) - 2: # Add an x between dimensions, e.g. 3x3
                output_shape_txt += "x"
            if ii == len(output_shape) - 2: # Add a newline between the last two dimensions, e.g. 3x3 \n 64
                output_shape_txt += "\n"

        # Add the name of the layer to the text, as a new line
        output_shape_txt += f"\n{layer.name}"

        # Return the text value and if it should be drawn above the layer
        return output_shape_txt, above
    
    def displayArchitecture(self,
                            dummy_layer_spacing=50,
                            show_dimension=True,
                            legend=False,
                            sizing_mode='balanced',
                            ignore_layers=[BatchNormalization,ReLU],
                            text_callable=text_callable,
                            save_to_file=None,
                            figsize=(8,4),
                            ax=None,
                            **kwargs
                            ):
        font_prop = font_manager.FontProperties(family='sans-serif',weight='bold')
        font_path = font_manager.findfont(font_prop)
        font = ImageFont.truetype(font_path,size=15)

        self.model.add(vk.SpacingDummyLayer(spacing=dummy_layer_spacing))

        if save_to_file:
            if type(save_to_file) == bool:
                save_to_file = f"{self.model.name}_architecture.png"

        
        if not ax:
            fig,ax = plt.subplots(1,1,figsize=figsize)

        img = vk.layered_view(self.model,
                        show_dimension=show_dimension,
                        legend=legend,
                        type_ignore=ignore_layers,
                        text_callable=text_callable,
                        sizing_mode=sizing_mode,
                        to_file=save_to_file,
                        font=font,
                        **kwargs
                        )

        ax.axis('off')
        ax.imshow(img)
        ax.set_title(f"Model Architecture: {self.model.name}\n")

def plotConfusionMatrix(y_true,y_pred,
                        true_labels=None,
                        predict_labels='same',
                        ax=None, show_legend=True,
                        title="Confusion Matrix",
                        title_padding=10,
                        ylabel="Actual",
                        xlabel="Predicted",
                        ax_label_padding=8,
                        color_dict = {'true' : "#89CC75",
                                      'false' : "#BE8686",
                                      'positive' : "#1900FF",
                                      'negative' : "#FFEE00"},
                        titleFontdict = {'fontsize': 15,
                                          'fontweight': 'bold'},
                        axlabelFontdict = {'fontsize': 12,
                                          'fontweight': 'bold'},
                        axtickFontdict = {'fontsize': 10,
                                          'fontweight': 'normal'},
                        axtextFontdict = {'fontsize': 15,
                                          'fontweight': 'normal'},
                        sum_classes = True,
                        sum_label = 'Total',
                        center_to_cm = False,
                        **kwargs):
    
    def parse_labels(display_labels):
        # Ensure that display_labels is an iterable numpy array (ndim >= 1)     
        if not display_labels:
            display_labels = np.unique(y_true).astype(str)
        elif display_labels in ['n2w','num2words','word_from_num','wordFromNum']:
            display_labels = np.unique(y_true).astype(int)
            if len(display_labels) == 2:
                display_labels = np.array(['Negative','Positive'])
            else:
                display_labels = np.array([num2words(i) for i in np.unique(y_true)])
        elif isinstance(display_labels,str) or not hasattr(display_labels,'__iter__'):
            display_labels = np.array(display_labels).reshape(-1)
        elif hasattr(display_labels,'__iter__'):
            display_labels = np.array(list(display_labels)).reshape(-1).astype(str)
        else:
            raise TypeError("Invalid display_labels")
        
        # Convert to strings.  In case of one-hot encoding, create negative labels with prefix "not_".
        if len(display_labels) > 1:
            display_labels = np.array(display_labels).astype(str)
        elif len(display_labels) == 1:
            display_labels = np.array([ 'not_' + str(i) for i in display_labels ] )
        else:
            raise ValueError(f"display_labels is of length {len(display_labels)}: {display_labels}")
        
        return display_labels
    
    true_labels = parse_labels(true_labels)
    if predict_labels == 'same':
        predict_labels = true_labels
    else:
        predict_labels = parse_labels(predict_labels)

    cm = confusion_matrix(y_true,y_pred)
    color_matrix = np.zeros(cm.shape)
    color_matrix[np.diag_indices_from(color_matrix)] = 1
    cmap = ListedColormap([color_dict['false'],color_dict['true']])

    if not ax:
        _,ax = plt.subplots(1,1, figsize=(4,4))

    if sum_classes:
        cm = np.vstack((cm,cm.sum(axis=0).reshape(1,-1)))
        cm = np.hstack((cm,cm.sum(axis=1).reshape(-1,1)))

        true_labels = np.append(true_labels,sum_label)
        predict_labels = np.append(predict_labels,sum_label)
        color_matrix = np.zeros(cm.shape)
        color_matrix[np.diag_indices_from(color_matrix)] = 1

        color_matrix[-1,:] = 2
        color_matrix[:,-1] = 2

        cmap = ListedColormap([
            color_dict['false'],
            color_dict['true'],
            '#f0f0f0'
        ])

        # Horizontal line separating main matrix from totals row
        ax.axhline(y=cm.shape[0] - 1.5, color='black', linewidth=2)
        # Vertical line separating main matrix from totals column
        ax.axvline(x=cm.shape[1] - 1.5, color='black', linewidth=2)

    if show_legend:
        handles = [plt.Line2D([0], [0], color=color_dict[key], lw=5) for key in color_dict.keys()]
        labels = list(color_dict.keys())
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    ax.set_title(title,
                 pad=title_padding,
                 fontdict=titleFontdict)

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_xticklabels(predict_labels,fontdict=axtickFontdict)
    ax.set_xlabel(xlabel,
                  labelpad=ax_label_padding,
                  fontdict=axlabelFontdict)

    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_yticklabels(true_labels,va='center',fontdict=axtickFontdict)
    ax.tick_params(axis='y',rotation=90)
    ax.set_ylabel(ylabel,
                  labelpad=ax_label_padding,
                  fontdict=axlabelFontdict)
    
    ax.imshow(color_matrix,cmap=cmap)

    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            text = str(cm[i,j])

            if sum_classes:
                # --- Distinguish Totals row/column ---
                is_total_row = (i == n_classes - 1)
                is_total_col = (j == n_classes - 1)

                if is_total_row or is_total_col:
                    # Style for Total cells (bold, different color, etc.)
                    fontd = {'fontsize': 12,
                            'fontweight': 'normal'}
                    text_color = "black"  # or color_dict['negative'] / custom
                else:
                    # Normal confusion matrix cell styling
                    fontd = axtextFontdict
                    text_color = color_dict['negative'] if i == 0 else color_dict['positive']

            else:
                # Normal confusion matrix cell styling
                fontd = axtextFontdict
                text_color = color_dict['negative'] if i == 0 else color_dict['positive']

            ax.text(i,j,text,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontdict=fontd
                    )
    # Draw thin gridlines between all cells
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)

    # Grid settings for minor ticks → thin light-colored gridlines
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Make sure the major tick marks don't show the grid
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')  # set the position of the x axis labe

    # After plotting matrix & before plt.show()
    n_rows, n_cols = cm.shape

    if sum_classes:
        true_n_rows = n_rows - 1
        true_n_cols = n_cols - 1
    else:
        true_n_rows = n_rows
        true_n_cols = n_cols

    if center_to_cm:

        # ----- Center X label over only main matrix -----
        # X-axis label to top but aligned to center of confusion matrix (not totals)
        center_x = true_n_cols / n_cols / 2  # normalized position
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(xlabel, fontdict=axlabelFontdict)
        ax.xaxis.set_label_coords(center_x,1.1)  # (x, y in axis coords)

        # ----- Center Y label over only main matrix -----
        center_y = 1 - true_n_rows / n_rows / 2  # normalized position
        ax.set_ylabel(ylabel, fontdict=axlabelFontdict)
        ax.yaxis.set_label_coords(-0.1, center_y)  # (x, y in axis coords)

    ax.tick_params(axis='both',which='both',length=0)
    return ax

### ======================================================================= ###
### =======================MODEL-EVALUATOR-CLASS=========================== ###
### ======================================================================= ###

class ModelEvaluator:
    def __init__(self, model, train_set, validation_set, test_set):
        self.model = model
        self.modelName = model.name

        _,y_train,X_train = yX_split(train_set)
        self.y_train = y_train
        self.X_train = X_train

        _,y_eval,X_eval = yX_split(test_set)
        self.y_true = y_eval
        self.X_test = X_eval

        _,y_valid,X_valid = yX_split(validation_set)
        self.y_valid = y_valid
        self.X_valid = X_valid
        
        # In case the model is already trained prior to instantiation of this class
        try:
            self.history = model.history
            self.epochs = model.history.epoch
            self.training_history = model.history.history
            self.y_prob = model.predict(self.X_test.load(), verbose=0)
            self.threshold = self.optimalThreshold()
            self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)
        except:
            self.history = None
            self.epochs = None
            self.training_history = None
            self.y_prob = None
            self.threshold = None
            self.y_pred = None
        
        self.saved_epoch_components = None
        self.summary_figure = None
        self.training_time = {}
            
    def __call__(self,*args,**kwargs):
        print(f"Instance called with args: {args}, kwargs: {kwargs}")
        return self

    def plotArchitecture(self, 
                         figsize=(12,8),
                         ax = None,
                         rankdir = 'TB',
                         features_to_show='all'):
        '''Wrapper method for plotting model architecture as a flowchart using Keras/TensorFlow's keras.utils.plot_model() in combination with various matplotlib.pyplot methods.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches.
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If not provided, a new figure and axis will be created.
        rankdir : {'LR', 'TB', 'BT', 'RL'}, optional
            The direction of the graph. 'LR' is left-to-right, 'TB' is top-to-bottom, 'BT' is bottom-to-top, and 'RL' is right-to-left.
        features_to_show : str or iterable
            A string or iterable container of strings containing one or more of the following:
            \n\tshow_shapes
            \n\tshow_dtype
            \n\tshow_layer_names
            \n\tshow_layer_activations
            \n\tshow_trainable

        Returns
        -------
        None
        '''
        
        model = self.model
        name = self.modelName
        suffixes = "shapes,dtype,layer_names,layer_activations,trainable".split(',')
        error=None
        if features_to_show == 'all':
            show_params = [f"show_{featname}" for featname in suffixes ]
        elif type(features_to_show) == str:
            featname = features_to_show.strip('show_')
            if featname in suffixes:
                show_params = [f"show_{featname}"]
            else:
                error=True
        elif isinstance(features_to_show,(list,tuple,np.ndarray,dict,set)):
            show_params = [f"show_{featname.strip('show_')}" for featname in features_to_show]
        else:
            error=True
    
        if error:
            raise ValueError(f"`features_to_show` must be either [1] a\
                                single string or [2] an iterable container of strings\
                                (e.g., list, tuple, numpy.ndarray)\
                                containing one or more of the following:\
                                \n{'\n\t'.join(suffixes)}\n")
    
        show_args = {k:True for k in show_params}
        filename = f"{name}_architecture.png"
        plot_model(model, to_file=filename, rankdir=rankdir, **show_args)
        print(f"\nSaved image to: \033[32m{filename}\033[0m\n")
    
        if not ax:
            fig,ax = plt.subplots(1,1,figsize=figsize)
            
        image = cv2.imread(filename)
        ax.imshow(image)
        _ = ax.set_title(name,fontsize=10)

    def vkPlotArchitecture(self,
                           show_dimension=True,
                           legend=True,
                           type_ignore=[BatchNormalization,ReLU],
                           sizing_mode='balanced',
                           ax=None,
                           figsize=(8,4)
                          ):
        '''Wrapper method for plotting model architecture as volumetric layers using using VisualKeras
        
        Parameters
        ----------
        show_dimension : bool, optional
            Whether to show the dimensions of each layer. The default is True.
        legend : bool, optional
            Whether to show the legend. The default is True.
        type_ignore : list, optional
            A list of layer types to ignore. The default is [BatchNormalization,ReLU].
        sizing_mode : str, optional
            The sizing mode for the layered view. The default is 'balanced'.
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If not provided, a new figure and axis will be created.
        figsize : tuple, optional
            The figure size in inches. The default is (8,4).

        Returns
        -------
        None
        '''

        model = self.model
        img = vk.layered_view(model=model,
                 show_dimension=show_dimension,
                 legend=legend,
                 type_ignore=type_ignore,
                 sizing_mode=sizing_mode
                )
        if not ax:
            fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(f"Model Architecture: {model.name}\n")
        
    def train(self, model=None,
              monitor='val_auc',
              patience=3,
              restore_best_weights=True,
              optimizer_function = Nadam,
              learning_rate = 1e-4,
              loss_function='binary_crossentropy',
              metrics=['accuracy',AUC(curve='ROC',name='auc')],
              num_epochs=100,
              batch_size=8,
              shuffle=False,
              verbose=1,
              live=False,
              **kwargs):

        '''
        Train the model using the provided hyperparameters.

        Parameters
        ----------
        model : keras.Model, optional
            The model to train. If not provided, the model passed to the class during initialization will be used.
        monitor : str, optional
            The metric to monitor during training. The default is 'val_auc'.
        patience : int, optional
            The number of epochs to wait before stopping training when the monitored metric has stopped improving. The default is 3.
        restore_best_weights : bool, optional
            Whether to restore the model's best weights when the monitored metric has stopped improving. The default is True.
        optimizer_function : keras.optimizers.Optimizer, optional
            The optimizer to use during training. The default is Nadam.
        learning_rate : float, optional
            The learning rate for the optimizer. The default is 1e-4.
        loss_function : str, optional
            The loss function to use during training. The default is 'binary_crossentropy'.
        metrics : list, optional
            A list of metrics to track during training. The default is ['accuracy',AUC(curve='ROC',name='auc')].
        num_epochs : int, optional
            The number of epochs to train the model for. The default is 100.
        batch_size : int, optional
            The batch size to use during training. The default is 8.
        shuffle : bool, optional
            Whether to shuffle the data during training. The default is False.
        verbose : int, optional
            The verbosity of the training process. The default is 1.
        **kwargs
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        None
        '''
        
        header_text = f"Training model: {self.model.name}"
        print(f"{'='*len(header_text)}\n{header_text}\n{'='*len(header_text)}")

        start = datetime.now()
        print(f"\n\033[32;1mBEGIN:\033[0m \033[100m{start.strftime('%Y-%m-%d %H:%M:%S')}\033[0m\n")
        print('━'*50,'\n')
        
        early_stop = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights
        )

        self.model.compile(
            optimizer = optimizer_function(learning_rate=learning_rate),
            loss = loss_function,
            metrics = metrics
        )

        callbacks = [early_stop]
        if live:
            live = LiveCapture(show_every='epoch')
            callbacks.append(live)

        self.history = self.model.fit(
            self.X_train.load(),
            self.y_train,
            validation_data=(self.X_valid.load(),self.y_valid),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=shuffle,
            verbose=verbose
        )

        self.training_history = self.history.history
        self.epochs = self.history.epoch

        end = datetime.now()
        print('━'*50)
        print(f"\n\033[31;1mEND:\033[0m \033[100m{end.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
        print('\n','-'*30)
        duration = end - start
        print(f"\033[33;1mELAPSED TIME:\033[0m \033[100m{duration}\033[0m\n")
        self.training_time = {
            'start': start,
            'end': end,
            'duration': duration
        }
    
    def optimalThreshold(self,method='roc_curve'):
        '''
        Calculate the optimal threshold for the model given the method.  This threshold is used to determine the binary class predictions (0 or 1) based on the predicted probabilities (float between 0 and 1, representing the probability of class 1).  For example, if the threshold is 0.5, then a predicted probability of 0.4 will be assigned a class of 0, and a predicted probability of 0.6 will be assigned a class of 1.

        This method should be called after the model has been trained.
        
        PARAMETERS
        ----------
        method : str, optional
            The method to use for calculating the optimal threshold. The default is 'roc_curve'.
            Supported methods are 'roc_curve' and 'accuracy_match'.

            Methods
            -------
            'roc_curve': 
                1. Generates FPR,TPR, and thresholds for the ROC curve using sklearn.metrics.roc_curve().
                2. Finds the best threshold to be the threshold that maximizes the difference between TPR and FPR.
                *Note: recall that Sensitivity = TPR = True Positives / (True Positives + False Negatives) and Specificity = TNR = 1 - FPR = True Negatives / (True Negatives + False Positives).  The ROC curve is a plot of TPR vs. FPR, which is the sensitivity vs. specificity in a binary classification problem.*
            'accuracy_match': accuracy maximization
                1. Calculates the proportion of positive class in y_true -- `true_rate`.
                2. Calculates the predicted proportion of positive class for a variety of thresholds -- `predRate`, calculated across 100 evenly-spaced thresholds between 0 and 1.
                3. Finds the threshold that minimizes the absolute difference between `true_rate` and `predRate`; i.e., the threshold that maximizes the accuracy of the predictions.

        
        Returns
        -------
        float
            The optimal threshold for the model given the method.
        '''
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_prob is None:
            self.y_prob = self.model.predict(self.X_test.load(), verbose=0)
        if method == 'roc_curve': 
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
            best_thresh = thresholds[np.argmax(tpr-fpr)]
            self.threshold = best_thresh
            return best_thresh
        elif method == 'accuracy_match':
            yPred = lambda threshold : np.where(self.y_prob > threshold, 1, 0)
            true_rate = self.y_true.sum() / len(self.y_true)
            predRate = lambda threshold: yPred(threshold).sum() / len(self.y_true)
            data={}
            for thresh in np.linspace(0,1,100):
                absdev = np.abs(true_rate - predRate(thresh))
                data[absdev] = thresh
            best_thresh = data[min(data.keys())]
            self.threshold = best_thresh
            return best_thresh

    def predict(self,threshold_method='roc_curve'):
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_prob is None:
            self.y_prob = self.model.predict(self.X_test.load(), verbose=0)
        
        if self.threshold is None:
            if threshold_method == 'roc_curve':
                self.threshold = self.optimalThreshold('roc_curve')
            elif threshold_method == 'accuracy_match':
                self.threshold = self.optimalThreshold('accuracy_match')
            else:
                print("Invalid threshold method -- defaulting to roc_curve method.")
                self.threshold = self.optimalThreshold()

        self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)
        
        return self.y_pred

    def plotCM(self,method='custom',ax=None,
               title='Confusion Matrix',
               figsize=(4,4),
               sum_classes=False,
               sum_label='Total',
               center_to_cm=True,
               show_legend=False,
               ):
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_pred is None:
            self.predict()

        if method == 'custom':
            ax=plotConfusionMatrix(self.y_true,self.y_pred,
                    title=title,
                    true_labels=['Normal','Glaucoma'],
                    predict_labels='n2w',
                    sum_classes=sum_classes,
                    center_to_cm=center_to_cm,
                    show_legend=show_legend,
                    sum_label=sum_label,
                    ax=ax,
                    figsize=figsize
                   )
        
        elif method == 'sklearn':
            cm = confusion_matrix(self.y_true,self.y_pred,labels=[0,1])
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, 
                display_labels=['Normal','Glaucoma']
            )
            if not ax:
                fig,ax = plt.subplots(1,1, figsize=figsize)
            cm_plot = disp.plot(values_format='d',
                            cmap=plt.cm.cividis,
                            colorbar=True,
                            ax=ax)
            ax.set_title(f"Confusion Matrix")
            return cm_plot
    
    def classificationReport(self,output_df=True,intro_text=None):
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_pred is None:
            self.predict()
        if intro_text:
            # print('-'*len(intro_text))
            print(f"\033[4m{intro_text}\033[0m")
        if output_df:
            cr = classification_report(self.y_true, self.y_pred, output_dict=True)
            df = pd.DataFrame(cr).T
            display(df)
            return df
        else:
            cr = classification_report(self.y_true, self.y_pred)
            print(cr)

    def getSavedEpoch(self):
        if self.training_history is None:
            raise RuntimeError("No history -- run train() first.")

        # Get saved epoch
        results = self.model.evaluate(self.X_valid.load(), self.y_valid, verbose=0)
        
        saved_loss, saved_acc, saved_auc = results

        # Find the epochs within the history that match the saved epoch within p precision
        get_matched_epochs = lambda p : [ i for i,v in enumerate(self.training_history['val_auc']) if round(v,p) == round(saved_auc,p) ]

        # Sometimes, the val_auc in history is a different precision than saved_auc, so we need to find the best match starting from higher precision and eventually falling back until we find a match or run out of epochs.  E.g., if saved_auc is 0.9999 and val_auc is 0.999, then the return value of get_matched_epochs(6) will be empty, but get_matched_epochs(3) will return the saved epoch as the only value in the list -- this is because the saved_auc will be rounded to 0.999 and the val_auc will be unchanged by the rounding and will be 0.999 --> round(0.999,3) == round(0.9999,3).
        saved_epoch = None
        for i in range(6):
            matched_epochs = get_matched_epochs(6-i)
            if len(matched_epochs) == 1:
                saved_epoch = matched_epochs[0]
                saved_auc = self.training_history['val_auc'][saved_epoch]
                break
        if not saved_epoch:
            if len(matched_epochs) > 1:
                raise RuntimeError(f"Multiple epochs have the same val_auc value: {matched_epochs}")
            else:
                raise RuntimeError("No epochs have the same val_auc value")
        
        self.saved_epoch_components = saved_epoch, saved_loss, saved_acc, saved_auc
        
        return saved_loss, saved_acc, saved_auc, saved_epoch
    
    def plotHistory(self,ax=None,figsize=(7.5,3),train_color='magenta',val_color='cyan'):
        if not self.saved_epoch_components:
            saved_loss, saved_acc, saved_auc, saved_epoch = self.getSavedEpoch()
        else:
            saved_loss, saved_acc, saved_auc, saved_epoch = self.saved_epoch_components   

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=figsize)
            
        ax.plot(self.training_history['val_auc'], label='val AUC', marker='.', color=val_color)
        ax.plot(self.training_history['auc'], label='train AUC', marker='.', color=train_color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.axvline(saved_epoch, linestyle='--', c='r', label='Saved Epoch')
        all_auc = self.training_history['auc'] + self.training_history['val_auc']
        all_auc = np.array(all_auc)
        midxax = all_auc.min() + ((all_auc.max() - all_auc.min()) / 2)
        ax.text(saved_epoch+.2, midxax, f"val:    {saved_auc:.4f}\ntrain:  {self.training_history['auc'][saved_epoch]:.4f}")
        ax.legend()
        ax.set_title('AUC over Epochs')
        plt.tight_layout()

        return fig,ax
    
    def plotROC(self,ax=None,use_test_set=False,figsize=(8,6)):
        if use_test_set:
            X = self.X_test
            y = self.y_true
        else:
            X = self.X_valid
            y = self.y_valid
        
        self.y_prob = self.model.predict(X.load(), verbose=0)

        roc_auc = roc_auc_score(y, self.y_prob)
        fpr, tpr, _ = roc_curve(y, self.y_prob)
        
        if self.threshold is None:
            self.threshold = self.optimalThreshold()

        if self.y_pred is None:
            self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)

        self.roc_auc = roc_auc

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=figsize)



        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % self.roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve',
                    fontsize=15,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', lw=0.5),
                    pad=22
                    )
        ax.legend(loc="lower right")

        btidx = np.argmax(tpr - fpr)
        btcoords = (fpr[btidx], tpr[btidx])
        ax.scatter(*btcoords,c='black',alpha=1,marker='o')
        btx,bty = btcoords
        # ax.text(btx+0.1,bty-0.15,'test')
        ax.annotate(xy=(btx+0.01,bty-0.01),
                    xytext=(btx+0.30,bty-0.45),
                    text=f'Best Threshold\n{self.threshold:.04f}\n(FPR,TPR)=\n({btx:.03f}, {bty:.03f})',
                    arrowprops = {'width':1,
                                'shrink':0.05
                                }
                )

    def summary_plots(self):
        self.plotHistory()
        fig,(l,r)=plt.subplots(1,2,figsize=(8,3))
        self.plotCM(ax=l,figsize=(4,4))
        self.plotROC(ax=r,figsize=(4,4))
        self.summary_figure = fig

    def strTimeDelta(self,duration:timedelta = None,delimiter=None,unitvers=0):

        if duration is None:
            if self.history is None:
                raise RuntimeError("No history -- run train() first.")
            duration = self.training_time['duration']

        years, remainder = divmod(duration.total_seconds(), 52 * 7 * 24 * 60**2 )
        # print('years: ',years,'rem: ', remainder)
        weeks, remainder = divmod(remainder,                     7 * 24 * 60**2 )
        # print('weeks: ',weeks,'rem: ', remainder)
        days, remainder = divmod(remainder,                          24 * 60**2 )
        # print('days: ', days,'rem: ', remainder)
        hours, remainder = divmod(remainder,                              60**2 )
        # print('hours: ', hours,'rem: ', remainder)
        minutes, remainder = divmod(remainder,                            60**1 )
        # print('minutes: ', minutes,'rem: ', remainder)
        seconds, remainder = divmod(remainder,                        1 )
        # print('seconds: ', seconds, 'rem: ', remainder)
        microseconds, remainder = divmod(remainder, 1e-6)
        # print('microseconds: ', microseconds)
        
        stringmap = {
            ('','y','yr','years'):years,
            ('','w','wk','weeks'):weeks,
            ('','d','day','days'):days,
            ('','h','hr','hours'):hours,
            ('','m','min','minutes'):minutes,
            ('','s','sec','seconds'):seconds,
            ('',u"\u03BCs", u"\u03BCsec",'microseconds'):microseconds
        }
        
        default_delimiters = {
            0 : ':',
            1 : ', ',
            2 : ', ',
            3 : ' | '
        }

        if not delimiter:
            delimiter = default_delimiters[unitvers]
        
        spacer = ' ' if unitvers > 0 else ''
        strings = []
        for k,v in stringmap.items():
            if v:
                s = f"{int(v)}{spacer}{k[unitvers]}"
                if v == 1 and len(k[unitvers]) > 1:
                    s = re.sub(r"s$",'',s)
                strings.append(s)
        
        result = delimiter.join(strings)
        if unitvers == 0:
            result = re.sub(delimiter+r"(\d+$)", '.'+r'\1', result)
        return result
    
    def evaluate(self):
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_prob is None:
            self.y_prob = self.model.predict(self.X_test.load(), verbose=0)
        if self.threshold is None:
            self.threshold = self.optimalThreshold()
        if self.y_pred is None:
            self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)

        header_text = f"Evaluating model: {self.model.name}"
        print(f"{'='*len(header_text)}\n{header_text}\n{'='*len(header_text)}")
        time_text = f"Training Time: \033[100m{self.strTimeDelta(unitvers=3,delimiter=', ')}\033[0m"
        print(f"\n{time_text}\n{'-'*(len(time_text)+3)}\n")
        self.summary_plots()
        self.classificationReport(intro_text="Classification Report")

    def train_and_evaluate(self,
                           monitor='val_auc',
                           patience=3,
                           restore_best_weights=True,
                           optimizer_function = Nadam,
                           learning_rate = 1e-4,
                           loss_function='binary_crossentropy',
                           metrics=['accuracy',AUC(curve='ROC',name='auc')],
                           num_epochs=100,
                           batch_size=8,
                           shuffle=False,
                           verbose=0,
                           show_architecture=False,
                           vk_show_dimension=True,
                           vk_legend=True,
                           vk_type_ignore=[BatchNormalization,ReLU],
                           vk_sizing_mode='balanced',
                           vk_figsize=(8,4),
                           **kwargs):
                           
        if show_architecture:
            modviz = ModelVisualizer(self.model)
            modviz.displayArchitecture(
                show_dimension=vk_show_dimension,
                legend=vk_legend,
                type_ignore=vk_type_ignore,
                sizing_mode=vk_sizing_mode,
                figsize=vk_figsize
                )
        
        self.train(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            optimizer_function = optimizer_function,
            learning_rate = learning_rate,
            loss_function=loss_function,
            metrics=metrics,
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbose,
            **kwargs
        )
        
        self.evaluate()



    