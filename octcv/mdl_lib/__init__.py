###=================================###
### OCTCV/octcv/mdl_lib/__init__.py ###
###=================================###

import os,sys

scriptDIR = os.path.dirname(os.path.abspath(__file__))
octcvDIR = os.path.dirname(scriptDIR)
projectDIR = os.path.dirname(octcvDIR)

sys.path.append(projectDIR)

from octcv.arrViz import *
from octcv.mdl_lib.callbacks import *

import numpy as np
import pandas as pd
import re
import glob
import os
import json
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import seaborn as sns
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
from keras.src.callbacks.history import History
from keras.models import load_model

from IPython.display import display,clear_output, Markdown,display_markdown
from IPython.display import HTML
import tensorflow as tf

import base64
from io import BytesIO
from PIL import Image

scriptPATH = os.path.realpath(__file__)
scriptDIR = os.path.dirname(scriptPATH)
projectDIR = os.path.dirname(scriptDIR)
modelingDIR = os.path.join(projectDIR,'p5_Modeling')

def numpy_to_html_img(arr, cmap_name='viridis', width='40%'):
    # 1. Convert numpy array to PIL Image
    # (Ensure your array is uint8; if it's float 0-1, multiply by 255)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    sm = plt.get_cmap(cmap_name)
    rgba_array = sm(arr)
    rgba_uint8 = (rgba_array * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8)
    
    # 2. Save to a bytes buffer
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    
    # 3. Encode to base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 4. Return the HTML img tag
    width = width if isinstance(width,str) else str(width)+'px'
    return f'<img src="data:image/png;base64,{img_str}" style="width:{width}; height:auto; align" />'

def describe_array(array):
    cols = "min mean median max std sum".split(' ')
    vals = [array.min(),array.mean(),np.median(array),array.max(),array.std(), array.sum()]
    return pd.DataFrame(vals,index=cols).T

def describeArrayHTML(array,title='',output='return',tblwidth=None):
    floatFormatter = lambda x : str(round(x,2)) if round(x) - x != 0 else f"{int(x):,d}"
    tid = np.random.randint(0,int(1e5))
    
    html = describe_array(array).to_html(
        index=False,
        border=0,
        justify='center',
        float_format=floatFormatter,
        table_id = tid
    )

    cmap = {
        'ttxt':'white',
        'tbkgr':'#585880',
        'htxt':'#29313d',
        'hbkgr':'#e0dfff',
        'ctxt':'black',
        'cbkgr':'white'
    }

    if tblwidth is None:
        tblwidth = 100 if title else 50
    
    styleTag = f"""
    <style>
      .tab-figure {{
        display: inline-block;
        border: 1px solid #ccc;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        margin: 10px;
      }}
      
      .tab-title {{
        background-color: {cmap['tbkgr']};
        color: {cmap['ttxt']};
        padding: 8px 8px;
        font-weight: bold;
        font-size: 100%;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        text-align: center;
      }}
      
        .dataframe[id="{tid}"] {{
            border-collapse: collapse;
            margin: 0 0 0 0;
            width: {tblwidth}%;
        }}
        
        .dataframe td, .dataframe th {{
            text-align: center !important;
            padding: 8px;
            border-color: #96D4D4 !important;
            overflow-wrap: hidden;
        }}
    
        .dataframe th {{
            color: {cmap['htxt']};
            background-color: {cmap['hbkgr']};
            border: 1px solid #5c6e89 !important;
        }}
    
        .dataframe td {{
            color: {cmap['ctxt']};
            background-color: {cmap['cbkgr']}; 
            border: 1px solid #96D4D4 !important; 
        }}
    </style>
    """

    htmlElements = [
        styleTag,
        '<div class="tab-figure">' if title else '',
         f'  <div class="tab-title">{title}</div>' if title else '',
        html,
        '  </div>' if title else '',
        '</div>' if title else ''
    ]
        
    htmlContent = '\n'.join(htmlElements)
    # print(htmlContent)
    match output:
        case 'html' | 'return':
            return htmlContent
        case 'disp':
            display(HTML(htmlContent))
        case _:
            display(HTML(htmlContent))

def imgStatsCompare(images,titles=('Original','Noisy')):
    cdiv = lambda innerHTML : f'<div style="text-align: center;">{innerHTML}</div>'
    itagL = cdiv(numpy_to_html_img(images[0]))
    itagR = cdiv(numpy_to_html_img(images[1]))
    tabL = cdiv(describeArrayHTML(images[0],title=titles[0],
                                  output='html',tblwidth=100))
    tabR = cdiv(describeArrayHTML(images[1],title=titles[1],
                                  output='html',tblwidth=100))
    html = f"""
    <table style="width: 100%; max-width: 800px; text-align: center;">
      <tbody style="text-align: center;">
        <tr style="text-align: center;">
          <td>{tabL}</td>
          <td>{tabR}</td>
        </tr>
        <tr style="text-align: center;">
        <div style="text-align: center;">
          <td>{itagL}</td>
          <td>{itagR}</td>
        </div>
        </tr>
      </tbody>
    </table>
    """
    display(HTML(html))
    
def MinMaxScaleNDArray(array,value_range=(0,255),verbose=False,histplots=False,preview=False):
    resmin,resmax = value_range
    # scaled = array - array.min() + resmin
    # scaled = scaled * (resmax / scaled.max())
    scaled = resmin + ((array - array.min()) / (array.max() - array.min())) * (resmax - resmin)
    scaled = scaled.astype(int)
    
    if verbose:
        array_stats = describe_array(array)
        scaled_stats = describe_array(scaled)
        compare = pd.concat([array_stats,scaled_stats],axis=0)
        compare.index = ['Original','MinMaxScaled']
        display(compare)
    if histplots and not preview:
        df = pd.DataFrame(np.array([array.flatten(),scaled.flatten()]).T,columns=['Original','Result'])
        # df.hist(figsize=(12,4),bins=100)
        plt.hist(df,alpha=0.5,bins=100)
    elif preview and histplots:
        fig,axes = plt.subplots(2,2,figsize=(10,6),sharex='col')
        
        # Histograms
        nbins = int( array.size / array.size**(2/3) )
        colors = ['green','red']
        data = [array.flatten(),scaled.flatten()]
        
        for i,ax in enumerate(axes[:,1]):
            ax.grid(True)
            sns.histplot(data[i],
                         kde=True,
                         line_kws=dict(
                             lw=2.5,
                             ls='-',
                             alpha=0.9
                         ),
                         color=colors[i], 
                         bins=nbins, 
                         alpha=0.5, 
                         ax=ax)
            # ax.hist(array.flatten(), color=colors[i], bins=nbins, alpha=0.9)
            ax.set_xlim(0,255)
            ax.set_xticks(np.arange(0,255,25))
            ax.set_ylabel('Number of Pixels',fontsize=12)
            if i==1:
                ax.set_xlabel('Pixel Intensity',fontsize=12)

        for i,ax in enumerate(axes[:,0]):
            ax.axis('off')

        # Images
        axes[0,0].imshow(array)
        axes[0,0].set_title('Original',fontsize=14)
        
        axes[1,0].imshow(scaled)
        axes[1,0].set_title('Result',fontsize=14)

        plt.tight_layout()
        
    elif preview and not histplots:
        fig,(l,r) = plt.subplots(1,2,figsize=(8,5))
        l.imshow(array); l.axis('off'); l.set_title('Original')
        r.imshow(scaled); r.axis('off'); r.set_title('Result')
        
    return scaled




class XVolSet:
    '''
    Class for storing image paths as X (predictor variables), while easily loading all images as arrays when needed via .load() method.
    '''
    def __init__(self,filepaths,default_load_normalized=True):
        self.filepaths = filepaths
        self.tf_dataset = None
        self.shape = filepaths.shape
        self.default_load_normalized = default_load_normalized
        self.summary_stats = None
        self.calculate_summary_stats(normalized=self.default_load_normalized, verbose=False)
        self.ndim = self.summary_stats['ndim']
        self.dtype = self.summary_stats['dtype']
        self.min = self.summary_stats['min']
        self.max = self.summary_stats['max']
        self.mean = self.summary_stats['mean']
        self.std = self.summary_stats['std']
        self.n_items = self.summary_stats['n_items']
        self.size = self.summary_stats['size']
        self.nbytes = self.summary_stats['nbytes']
    
    def __getitem__(self, index):
        # Keep columns intact
        if isinstance(index, int):
            sub_df = self.filepaths.iloc[[index]]  # keep as DataFrame, 1 row
        else:
            sub_df = self.filepaths.iloc[index]    # slice
        return XVolSet(filepaths=sub_df,default_load_normalized=self.default_load_normalized)

    def __len__(self):
        return self.shape[0]
    
    def getDisplayFilePaths(self):
        # If 2D array/DataFrame:
        if self.filepaths.ndim == 2:
            # If there are 2 columns
            if self.filepaths.shape[1] == 2:
                # Get the second column only
                if isinstance(self.filepaths,pd.DataFrame):
                    filepaths = self.filepaths.iloc[:,1].values
                else:
                    filepaths = self.filepaths[1]
            # if there is only 1 column
            elif self.filepaths.shape[1] == 1:
                # Get the only column as a linear array
                if isinstance(self.filepaths,pd.DataFrame):
                    filepaths = self.filepaths.iloc[:,0].values
                else:
                    filepaths = self.filepaths[0]
            else:
                raise ValueError('Improper self.filepaths shape')
        # If 1D linear array
        elif self.filepaths.ndim == 1:
            filepaths = self.filepaths
        else:
            raise ValueError('Improper self.filepaths shape')
        return filepaths

    def tfBatchedDataset(self):
        '''Converts filepaths to non-normalized volumes into a tensorflow dataset for calculation of summary statistics in batches rather than all at once.  This prevents running out of RAM in the case of large datasets.

        The method stores the resulting tensorflow dataset under self.tf_dataset if not already done, and subsequent runs will simply return self.tf_dataset.
        '''
        if self.tf_dataset is None:
            filepaths = self.getDisplayFilePaths()
            
            def load_npy(path):
                data = np.load(path.numpy().decode())
                return data.astype(np.float32)
            
            tf_dataset = tf.data.Dataset.from_tensor_slices(filepaths)
            
            tf_dataset = tf_dataset.map(
                lambda x:tf.py_function(load_npy, [x], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
            self.tf_dataset = tf_dataset
        else:
            tf_dataset = self.tf_dataset
        return tf_dataset
        
    def getMax(self):
        tf_dataset = self.tfBatchedDataset()
        global_max = -np.inf
        for batch in tf_dataset:
            batch_max = tf.reduce_max(batch)
            if batch_max > global_max:
                global_max = batch_max
        return global_max.numpy()
        
    def getMin(self):
        tf_dataset = self.tfBatchedDataset()
        global_min = np.inf
        for batch in tf_dataset:
            batch_min = tf.reduce_min(batch)
            if batch_min < global_min:
                global_min = batch_min
        return global_min.numpy()

    def getMean(self):
        tf_dataset = self.tfBatchedDataset()
        batch_means = []
        for batch in tf_dataset:
            batch_means.append(tf.reduce_mean(batch).numpy())
        return np.array(batch_means).mean()
        
    def getStDev(self):
        tf_dataset = self.tfBatchedDataset()
        all_means = []
        all_vars = []
        all_sizes = []
        
        # 1. Collect statistics for each batch
        for batch in tf_dataset:
            all_means.append(tf.reduce_mean(batch).numpy())
            all_vars.append(tf.math.reduce_variance(batch).numpy())
            all_sizes.append(tf.size(batch).numpy())
        
        n_total = sum(all_sizes)
        
        # 2. Calculate the combined mean
        mean_total = sum(m * s for m, s in zip(all_means, all_sizes)) / n_total
        
        # 3. Calculate "Within-Batch" Variance (weighted average of batch variances)
        var_within = sum(v * s for v, s in zip(all_vars, all_sizes)) / n_total
        
        # 4. Calculate "Between-Batch" Variance (how far batch means are from total mean)
        var_between = sum(s * (m - mean_total)**2 for m, s in zip(all_means, all_sizes)) / n_total
        
        # 5. Total Standard Deviation is the square root of the sum of variances
        return np.sqrt(var_within + var_between)

    def getSize(self):
        filepaths = self.getDisplayFilePaths()
        
        total_size = 0
        for path in filepaths:
            total_size += os.path.getsize(path)

        hrbytes = int(total_size)
        if hrbytes > 1e9:
            hrbytes = f"{hrbytes / 1e9:.2f} GB"
        elif hrbytes > 1e6:
            hrbytes = f"{hrbytes / 1e6:.2f} MB"
        elif hrbytes > 1e3:
            hrbytes = f"{hrbytes / 1e3:.2f} kB"
        else:
            hrbytes = f"{hrbytes} bytes"

        return total_size,hrbytes

    def getShapeNDimsNitemsDtype(self):
        '''Calculate shape,ndim,size of final dataset without loading the whole thing.

        RETURNS
        ---
        tuple (final_shape, final_ndims, final_nitems)
        '''
        # safe to assume all data points (visual arrays like images/volumes) would have the same shape in a given dataset, as the CNN only accepts one input shape; thus can sample any random volume, load it, and simply add a batch and channel dimension.  Batch dim is multiplied by the number of datapoints.
        filepaths = self.getDisplayFilePaths()
        rvpath = np.random.choice(filepaths)
        ranvol = np.load(rvpath)
        baseshape = list(ranvol.shape)
        N = len(filepaths)

        dtype = ranvol.dtype
        
        final_shape = np.array([N] + baseshape + [0])
        
        final_ndims = len(final_shape)

        # Multiply non-zero elements of a single volume, then multiply that by number of volumes N
        final_nitems = N * np.prod(final_shape[final_shape != 0])

        # Return tuple of global dataset shape, global dataset number of dimensions, and total number of elements
        return tuple(final_shape),final_ndims,final_nitems,dtype
    
    def calculate_summary_stats(self,normalized=None, verbose=True):
        # volumes = self.load(normalized=normalized) 
        tf_dataset = self.tfBatchedDataset()
        
        size,hrbytes = self.getSize()
        shape,ndim,nitems,dtype = self.getShapeNDimsNitemsDtype()
        
        self.summary_stats = {
            'min': self.getMin(),
            'max': self.getMax(),
            'mean': self.getMean(),
            'std': self.getStDev(),
            'shape': str(shape),
            'dtype': dtype,
            'ndim': ndim,
            'n_items': f"{nitems:,}",
            'size' : size,
            'nbytes': hrbytes
        }
        if verbose:
            self.describe()
    
    def describe(self):
        summary_stats = self.summary_stats
        sumdf = pd.DataFrame(summary_stats,index=['value'])
        vals = sumdf.T['value'].apply(lambda x : f"{round(x,2):.2f}" if isinstance(x,float) else x)
        data = [[v] for v in vals]
        sumdf = pd.DataFrame(data,index=sumdf.columns).T
        display(sumdf)

    def load(self, normalized=None):
        if normalized is None:
            normalized = self.default_load_normalized

        image_paths = self.filepaths
    
        # Ensure consistent DataFrame
        if isinstance(image_paths, pd.Series):
            image_paths = image_paths.to_frame().T
    
        nrows, ncols = image_paths.shape
    
        if ncols == 2:
            col_name = 'normalized_array' if normalized else [c for c in image_paths.columns if c.startswith('display')][0]
            xarrs = image_paths[col_name].apply(vizInputParser).values
            xarrs = np.stack(xarrs, axis=0)
        elif ncols == 1:
            xarrs = np.array([vizInputParser(path) for path in image_paths.iloc[:,0]])  # note .iloc[:,0] for 1-column DataFrame
        else:
            raise ValueError(f"Unsupported number of columns {ncols} in XVolSet")
    
        xarrs = xarrs[..., np.newaxis]
        return xarrs
    
    def max(self, normalized=None):
        return self.load(normalized=normalized).max()
    
    def min(self, normalized=None):
        return self.load(normalized=normalized).min()
    
    def __repr__(self):
        
        return f"XVolSet({self.shape[0]})"


# Function to split datasets into y,X; but X is the XVolSet instance
def yX_split(df,
             one_hot_drop=True,
             summary_report=False,
             default_load_normalized=True,
             display_filepath_columns=None
            ):
    '''
    Splits datasets into y and X, along with y_labels (strings) for downstream interpretation.

    PARAMS:
    -------
    df    :    DataFrame, formated per OCTCV project pre-processing
    one_hot_drop: bool, if True, will drop extra column in y
    summary_report : bool, if True, will print a summary of return value stats
    default_load_normalized : bool, if True, the .load() method of X (XVolSet instance) will by default use the paths to the normalized volume array (if it exists) instead of the display_volume / original paths.  Default is True.
    display_filepath_columns : list,str,numpy.ndarray, or None -- collection of column names, or str if single column, with filepaths to the volume paths to include in the resulting XVolSet instance; if None, will assume a default display filepath of "display_volume".
    ***NOTE: will attempt to include ['normalized_array'] column in display_filepath_columns as well, but otherwise omit this if it does not exist.***

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
    if display_filepath_columns is None:
        disp_cols = [ c for c in df.columns if c.startswith('display') ][0] # can be useful for visualization later on
        disp_cols = [disp_cols]
    elif isinstance(display_filepath_columns,str):
        disp_cols = [display_filepath_columns]
    elif isinstance(display_filepath_columns,(list,tuple,np.ndarray)):
        disp_cols = display_filepath_columns
    else:
        raise ValueError(f"Invalid type for display_filepath_columns: {type(display_filepath_columns)} -- acceptable types include list,tuple, numpy.ndarray, str, and None.")
        
    try:
        X_paths = df[['normalized_array'] + list(disp_cols)] # "normalized_array" part for training / inference
    except:
        X_paths = df[list(disp_cols)]
        
    X = XVolSet(X_paths,default_load_normalized=default_load_normalized)

    if summary_report:
        print(f"y_labels shape: {y_labels.shape}")
        print(f"y shape: {y.shape}")
        print(f"X shape {X.shape}")
        
    return y_labels, y, X


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

def plotConfusionMatrix(y_true=None,y_pred=None,
                        cm_array=None,
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
    
    # Parse class labels for display as tick labels in plot
    # The prediction labels can either be the same (default) as the true labels or specified by the user
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
    
    # Parse data inputs - either directly inputting the confusion matrix as a numpy array or providing y_true and y_pred so that the confusion matrix can be calculated

    if np.array(cm_array).size >= 4:
        cm = np.array(cm_array)
    elif np.array(y_true).size > 0 and np.array(y_pred).size > 0:
        cm = confusion_matrix(y_true,y_pred)
    else:
        raise ValueError(f"Must provide either:\n\t*[1] both `y_true` and `y_pred`\n\tor\n\t*[2] `cm_array` (size >= 4)\n\nReceived:\n\ty_true: {y_true}\n\ty_pred: {y_pred}\n\tcm_array: {cm_array}\n")
    
    # Create color mapping to distinguish true (values on the diagonal of the confusion matrix) and false predictions (all other values).  Colors assigned to true and false predictions can be changed in color_dict -- default is "#89CC75" (a light shade of green) for True (TP and TN in the case of a binary confusion matrix) and "#BE8686" (a light shade of red) for False (FP and FN in the case of a binary confusion matrix).
    color_matrix = np.zeros(cm.shape)
    color_matrix[np.diag_indices_from(color_matrix)] = 1
    cmap = ListedColormap([color_dict['false'],color_dict['true']])

    if not ax:
        _,ax = plt.subplots(1,1, figsize=(4,4))

    # If sum_classes is True, the confusion matrix will include a row and column for the totals - i.e., an additional row at the bottom containing the sum of each column (totals for each predicted label) and an additional column to the right containing the sum of each row (totals for each true label), along with the grand total (total number of data points used to produce the confusion matrix - i.e., len(y_true)=len(y_pred), which should be the number of data points in the test/evaluation dataset ) in the bottom right corner.
    if sum_classes:
        # Stack the additional row (totals for predicted labels) at the bottom of the original confusion matrix
        cm = np.vstack((cm,cm.sum(axis=0).reshape(1,-1)))
        # Stack the additional column (totals for true labels) to the right of the original confusion matrix
        cm = np.hstack((cm,cm.sum(axis=1).reshape(-1,1)))

        # Append the sum_label to the true_labels and predict_labels arrays
        true_labels = np.append(true_labels,sum_label)
        predict_labels = np.append(predict_labels,sum_label)

        # Update the color matrix to match the new shape while preserving the color coding for true and false predictions (diagonal elements are true predictions, off-diagonal elements are false predictions).
        color_matrix = np.zeros(cm.shape)
        color_matrix[np.diag_indices_from(color_matrix)] = 1

        # Distinguish the newly added row and column color from those of the actual confusion matrix
        color_matrix[-1,:] = 2
        color_matrix[:,-1] = 2

        cmap = ListedColormap([
            color_dict['false'], # i.e., all 0's in color_matrix
            color_dict['true'], # i.e., all 1's in color_matrix
            '#f0f0f0' # i.e., all 2's in color_matrix
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

    # Add text for the values of each cell of the confusion matrix
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
                    text_color = color_dict['negative'] if j == 0 else color_dict['positive']

            else:
                # Normal confusion matrix cell styling
                fontd = axtextFontdict
                text_color = color_dict['negative'] if j == 0 else color_dict['positive']

            ax.text(j,i,text,
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
    return ax,cm

### ======================================================================= ###
### =======================MODEL-EVALUATOR-CLASS=========================== ###
### ======================================================================= ###

class ModelEvaluator:
    def __init__(self, model, train_set, validation_set, test_set, model_name=None):
        self.model = model

        if model_name:
            model.name = model_name

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
            self.roc_data = self.generateROCData()
            self.roc_auc = self.roc_data['auc']
            self.classification_report = self.classificationReport()
        except:
            self.history = None
            self.epochs = None
            self.training_history = None
            self.y_prob = None
            self.threshold = None
            self.y_pred = None
            self.roc_auc = None
            self.roc_data = None
            self.classification_report = None
            self.metrics = None
        
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
                           figsize=(8,4),
                           save_to_file=False,
                           output_path=None,
                           output_filename=None,
                           output_dir=None
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
        if save_to_file:
            if not (output_path or output_filename or output_dir):
                raise ValueError("At least one of `output_path`, `output_filename`, or `output_dir` must be provided -- except if `output_path` is provided, both `output_filename` and `output_dir` must be None (i.e., exactly one argument must be provided).")
            if output_path and not (output_filename or output_dir):
                oPATH = output_path
            elif output_filename and not (output_path or output_dir):
                oDIR = os.path.join(modelingDIR, 'models', self.model.name)
                os.makedirs(oDIR,exist_ok=True)
                oNAME = output_filename
                oPATH = os.path.join(oDIR,oNAME)
            elif output_dir and not (output_path or output_filename):
                oDIR = output_dir
                os.makedirs(oDIR,exist_ok=True)
                oNAME = f"{model.name}_VK-architecture.png"
                oPATH = os.path.join(oDIR,oNAME)
            else:
                raise ValueError("Ambiguous path for saving image; if `output_path` is provided, both `output_filename` and `output_dir` must be None.") 
            plt.savefig(oPATH)
            print(f"\nSaved image to: \033[32m{os.path.relpath(oPATH)}\033[0m\n")
        
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
              live_plot=False,
              metrics_to_plot=['auc','val_auc'],
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
        start = datetime.now()
        if not live_plot:
            header_text = f"Training model: {self.model.name}"
            print(f"{'='*len(header_text)}\n{header_text}\n{'='*len(header_text)}")
            print(f"\n\033[32;1mBEGIN:\033[0m \033[100;97m{start.strftime('%Y-%m-%d %H:%M:%S')}\033[0m\n")
            print('━'*50,'\n')
        


        self.model.compile(
            optimizer = optimizer_function(learning_rate=learning_rate),
            loss = loss_function,
            metrics = metrics
        )

        # CALLBACKS
        callbacks = []
        early_stop = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
        callbacks.append(early_stop)

        if live_plot:
            callbacks.append(LivePlot(metrics_to_plot=metrics_to_plot))

        if verbose == 0:
            callbacks.append(EpochProgressBar())

        # RUN TRAINING and STORE HISTORY
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
        duration = end - start
        self.training_time = {
            'start': start,
            'end': end,
            'duration': duration
        }

        if not live_plot:
            print('━'*50)
            print(f"\n\033[31;1mEND:\033[0m \033[100;97m{end.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
            print('\n','-'*30)
            print(f"\033[33;1mELAPSED TIME:\033[0m \033[100;97m{duration}\033[0m\n")

    
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
               save_to_file=False
               ):
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_pred is None:
            self.predict()

        if method == 'custom':
            ax,cm = plotConfusionMatrix(self.y_true,
                                        self.y_pred,
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
            selfcmarr = np.array(self.confusion_matrix)
            if not selfcmarr.any() or selfcmarr.size < 4:
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

        self.confusion_matrix = cm
            
        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, 'models', self.model.name, f"{self.model.name}_confusion_matrix.png")
            saveDIR = os.path.dirname(savePATH)
            os.makedirs(saveDIR, exist_ok=True)
            plt.savefig(savePATH)
    
    def classificationReport(self,
                             display_format='df',
                             intro_text=None,
                             save_to_file=False
                             ):
        if not self.classification_report:

            if self.history is None:
                raise RuntimeError("No history -- run train() first.")
            if self.y_pred is None:
                self.predict()
            if intro_text:
                # print('-'*len(intro_text))
                print(f"\033[4m{intro_text}\033[0m")

            cr = classification_report(self.y_true, self.y_pred, output_dict=True)
            self.classification_report = {'dict':cr}
            cr = pd.DataFrame(cr).T
            self.classification_report['df'] = cr        
            cr = classification_report(self.y_true, self.y_pred)
            self.classification_report['text'] = cr

        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, 'models', self.model.name, f"{self.model.name}_classification_report.csv")
            
            saveDIR = os.path.dirname(savePATH)
            os.makedirs(saveDIR, exist_ok=True)

            self.classification_report['df'].to_csv(savePATH, index=False)

        if display_format == 'df':
            display(self.classification_report['df'])
        elif display_format == 'text':
            print(self.classification_report['text'])
        elif display_format == 'dict':
            print(self.classification_report['dict'])

        return self.classification_report
            
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
                raise RuntimeError(f"Multiple epochs have the same val_auc value: {matched_epochs} -- selecting epoch {matched_epochs[-1]}")
            else:
                raise RuntimeError("No epochs have the same val_auc value")

        self.saved_epoch_components = saved_epoch, saved_loss, saved_acc, saved_auc
        
        return saved_loss, saved_acc, saved_auc, saved_epoch
    
    def plotHistory(self,ax=None,figsize=(7.5,3),
                    train_color='magenta',val_color='cyan',
                    save_to_file=False
                    ):
        if not self.saved_epoch_components:
            try:
                saved_loss, saved_acc, saved_auc, saved_epoch = self.getSavedEpoch()
            except Exception as e:
                saved_auc, saved_epoch = None, None
        else:
            saved_loss, saved_acc, saved_auc, saved_epoch = self.saved_epoch_components   

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=figsize)
            
        ax.plot(self.training_history['val_auc'], label='val AUC', marker='.', color=val_color)
        ax.plot(self.training_history['auc'], label='train AUC', marker='.', color=train_color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')

        if saved_epoch:
            ax.axvline(saved_epoch, linestyle='--', c='r', label='Saved Epoch')
            all_auc = self.training_history['auc'] + self.training_history['val_auc']
            all_auc = np.array(all_auc)
            midxax = all_auc.min() + ((all_auc.max() - all_auc.min()) / 2)
            ax.text(saved_epoch+.2, midxax, f"val:    {saved_auc:.4f}\ntrain:  {self.training_history['auc'][saved_epoch]:.4f}")
        ax.legend()

        ax.set_title('AUC over Epochs')

        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, 'models',  self.model.name, f"{self.model.name}_history.png")
            saveDIR = os.path.dirname(savePATH)
            os.makedirs(saveDIR, exist_ok=True)
            plt.savefig(savePATH)

        return fig,ax
    
    def generateROCData(self,
                        use_test_set=False,
                        save_to_file=False
                        ):
        if use_test_set:
            X = self.X_test
            y = self.y_true
        else:
            X = self.X_valid
            y = self.y_valid
        
        if np.array(self.y_prob).any():
            self.y_prob = self.model.predict(X.load(), verbose=0)

        roc_auc = roc_auc_score(y, self.y_prob)
        fpr, tpr, thresholds = roc_curve(y, self.y_prob)
        roc_data = {
            'auc':roc_auc,
            'data': {
                'fpr':fpr,
                'tpr':tpr,
                'thresholds':thresholds
                }
        }
        
        self.roc_data = roc_data

        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, 'models',self.model.name, f"{self.model.name}_roc_data.json")
            saveDIR = os.path.dirname(savePATH)
            os.makedirs(saveDIR, exist_ok=True)
            with open(savePATH,'w') as f:
                json.dump(roc_data,f)

        return roc_data
    
    def plotROC(self,ax=None,
                use_test_set=False,
                figsize=(8,6),
                save_to_file=False
                ):
        
        if not self.roc_data:
            roc_data = self.generateROCData(use_test_set=use_test_set)
            self.roc_data = roc_data
        else:
            roc_data = self.roc_data

        fpr = roc_data['data']['fpr']
        tpr = roc_data['data']['tpr']
        roc_auc = roc_data['auc']
        self.roc_auc = roc_auc

        if self.threshold is None:
            self.threshold = self.optimalThreshold()

        if self.y_pred is None:
            self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)

        if not ax:
            _,ax = plt.subplots(1,1, figsize=figsize)

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

        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, 'models', self.model.name, f"{self.model.name}_ROC.png")
            saveDIR = os.path.dirname(savePATH)
            os.makedirs(saveDIR, exist_ok=True)
            plt.savefig(savePATH)

        return ax

    def summary_plots(self,save_to_file=False,save_individual_plots=False):
        self.plotHistory(save_to_file=save_individual_plots)

        fig,(l,r)=plt.subplots(1,2,figsize=(8,3))

        self.plotCM(ax=l,figsize=(4,4),
                    save_to_file=save_individual_plots)
        
        self.plotROC(ax=r,figsize=(4,4),
                     save_to_file=save_individual_plots)
        
        self.summary_figure = fig

        if save_to_file:
            if isinstance(save_to_file, str):
                savePATH = save_to_file
            else:
                savePATH = os.path.join(modelingDIR, self.model.name, f"{self.model.name}_summary-plots.png")
            plt.savefig(savePATH)

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
    
    def evaluate(self,
                 save_performance_metrics=False,
                 savePATH=None,
                 saveDIR=None,
                 saveFILENAME=None,
                 verbose=0
                 ):
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
        time_text = f"Training Time: \033[100;97m{self.strTimeDelta(unitvers=3,delimiter=', ')}\033[0m"
        print(f"\n{time_text}\n{'-'*(len(time_text)+3)}\n")
        self.summary_plots()
        self.classificationReport(intro_text="Classification Report")

        if save_performance_metrics:
            if savePATH and not (saveDIR or saveFILENAME):
                saveDIR = os.path.dirname(savePATH)
                saveFILENAME = os.path.basename(savePATH)
            elif saveFILENAME and not (saveDIR or savePATH):
                saveDIR = os.path.join(modelingDIR, self.model.name)
            elif saveDIR and not (saveFILENAME or savePATH):
                saveFILENAME = f"{self.model.name}_metrics.json"
            elif saveDIR and saveFILENAME:
                pass
            elif savePATH and not (saveDIR and saveFILENAME):
                raise ValueError("Path for saving performance metrics is ambiguous -- if savePATH is provided, leave both saveDIR and saveFILENAME as None.")
            else:
                saveDIR = os.path.join(modelingDIR,'models', self.model.name)
                os.makedirs(saveDIR, exist_ok=True)
                saveFILENAME = f"{self.model.name}_metrics.json"
            self.savePerformanceMetrics(filename=saveFILENAME,basepath=saveDIR)
            if int(verbose) > 0:
                savePATH = os.path.join(saveDIR,saveFILENAME)
                print(f"\nPerformance metrics saved to: {os.path.relpath(savePATH)}\n")

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
                           save_performance_metrics=False,
                           savePATH=None,
                           saveDIR=None,
                           saveFILENAME=None,
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
        
        complete = False
        j=0
        while not complete:
            try:
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
                
                self.evaluate(
                    save_performance_metrics=save_performance_metrics,
                    savePATH=savePATH,
                    saveDIR=saveDIR,
                    saveFILENAME=saveFILENAME,
                    verbose=verbose
                )
                complete = True
            except KeyboardInterrupt:
                break
            except:
                complete = False
            j += 1
            if j > 10:
                break

    def generatePerformanceMetrics(self):
            if self.history is None:
                raise RuntimeError("No history -- run train() first.")
            if self.training_history is None:
                self.training_history = self.history.history
            if self.y_true is None:
                _,y_eval,X_eval = yX_split(self.test_set)
                self.y_true = y_eval
            if self.y_pred is None:
                self.y_pred = self.predict()
            if self.confusion_matrix is None:
                self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
            if self.roc_data is None:
                self.roc_data = self.generateROCData()
            if self.classification_report is None:
                self.classification_report = self.classificationReport(display_format=None)

            training_datetimes = self.training_time
            training_seconds = {}
            for k,v in training_datetimes.items():
                if isinstance(v,timedelta):
                    training_seconds[k] = v.total_seconds()
                elif isinstance(v,datetime):
                    training_seconds[k] = v.timestamp()

            metrics = {
                'name': self.model.name,
                'n_params': self.model.count_params(),
                'n_layers': len(self.model.layers),
                'time': training_seconds,
                'history': self.training_history,
                'cm': self.confusion_matrix.tolist(),
                'threshold' : self.threshold,
                'roc' : self.roc_data,
                'report' : self.classification_report['dict']
            }
            self.metrics = metrics
            return metrics

    def savePerformanceMetrics(self, filename=None,basepath=None):
        if not self.metrics:
            metrics = self.generatePerformanceMetrics()
        else:
            metrics = self.metrics
            
        metrics = makeJSONserializable(metrics)

        if not filename:
            filename = f"{self.model.name}_metrics.json"
        if not basepath:
            basepath = os.path.join(modelingDIR, 'models', self.model.name)

        os.makedirs(basepath, exist_ok=True)
        filePATH = os.path.join(basepath, filename)
        with open(filePATH, 'w') as f:
            json.dump(metrics, f)

    # def save(self):
    #     if self.history is None:
    #         raise RuntimeError("No history -- run train() first.")
        
    #     basedir = './models'
    #     mdldir = os.path.join(basedir, self.model.name)
    #     os.makedirs(mdldir, exist_ok=True)

    #     wtpath = os.path.join(mdldir, f"{self.model.name}.h5")
    #     archpath = os.path.join(mdldir, f"{self.model.name}.json")
    #     hist
    #     filename = os.path.join(wtdir, f"{self.model.name}.h5")
    #     self.model.save(filename)
    #     print(f"Model saved to: {filename}")

    # def load(self,basepath):
    #     mdlpath = glob.glob(os.path.join(basepath, '*.keras'))[0]
    #     self.model = load_model(mdlpath)

    #     histpath = glob.glob(os.path.join(basepath, '*_history.csv'))[0]
    #     histdf = pd.read_csv(histpath)
    #     self.training_history{ k:list(v) for k,v in dict(histdf).items() }
    #     self.model.history = History()
    #     self.model.history.history = self.training_history
    #     self.history = self.model.history

    #     print(f"Model loaded from: {mdlpath}")
    #     print(f"History loaded from: {histpath}")

   
def convert_float32_to_float(data):
    """
    Recursively converts all numpy.float32 values in a nested dictionary
    or list to standard Python floats.
    """
    if isinstance(data, dict):
        return {k: convert_float32_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_float32_to_float(elem) for elem in data]
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data
    
def convert_numpyarray_to_list(data):
    if isinstance(data, dict):
        return {k: convert_numpyarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpyarray_to_list(elem) for elem in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
    
def makeJSONserializable(data):
    data = convert_numpyarray_to_list(data)
    data = convert_float32_to_float(data)
    return data

# class ModelComparison:
#     def __init__(self, model_names: list):
#         self.models = model_names
#         for model in self.models:
#             if os.path.

    