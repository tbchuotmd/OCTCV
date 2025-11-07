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
from keras.utils import plot_model
from sklearn.metrics import roc_curve, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import visualkeras as vk
from PIL import ImageFont
from matplotlib import font_manager
import cv2

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


class ModelEvaluator:
    # class CustomCallback(Callback):
    #     def __init__(self):
    #         super(CustomCallback, self).__init__()
    #         self.batch = 0
    #         self.total_batches = 0
    
    #     def on_train_batch_begin(self, batch, logs=None):
    #         self.batch += 1
    #         self.total_batches = logs.get('batch') + 1
    
    #     def on_train_batch_end(self, batch, logs=None):
    #         # Customize the output here
    #         output = f'Epoch {self.model.epoch}, Batch {batch}/{self.total_batches}'
    #         print(output)
    
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
            
    def __call__(self,*args,**kwargs):
        print(f"Instance called with args: {args}, kwargs: {kwargs}")
        return self

    def plotArchitecture(self, 
                         figsize=(12,8),
                         ax = None,
                         rankdir = 'TB',
                         features_to_show='all'):
        '''Wrapper method for plotting model architecture as a flowchart using Keras/TensorFlow's keras.utils.plot_model() in combination with various matplotlib.pyplot methods.'''
        
        model = self.model
        name = self.modelName
        suffixes = "shapes,dtype,layer_names,layer_activations,trainable".split(',')
        error=None
        if features_to_show == 'all':
            show_params = [f"show_{featname}" for featname in suffixes ]
        elif type(features_to_show) == str:
            featname = features_to_show.strip('show_')
            if feature in suffixes:
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
        '''Wrapper method for plotting model architecture as volumetric layers using using VisualKeras'''

        model = self.model
        img = layered_view(model=model,
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
              **kwargs):

        print(f"{'='*30}\nTraining model: {self.model.name}\n{'='*30}\n") 
        
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

        self.history = self.model.fit(
            self.X_train.load(),
            self.y_train,
            validation_data=(self.X_valid.load(),self.y_valid),
            epochs=num_epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            shuffle=shuffle,
            verbose=verbose
        )

        self.training_history = self.history.history
        self.epochs = self.history.epoch
    
    def optimalThreshold(self,method='roc_curve'):
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

    def plotCM(self,ax=None):
        
        if self.history is None:
            raise RuntimeError("No history -- run train() first.")
        if self.y_pred is None:
            self.predict()

        cm = confusion_matrix(self.y_true,self.y_pred,labels=[0,1])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=['Normal','Glaucoma']
        )

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=(4,4))

        cm_plot = disp.plot(values_format='d',
                         cmap=plt.cm.cividis,
                         colorbar=True,
                         ax=ax)
        
        ax.set_title(f"Confusion Matrix")
    
    def plotHistory(self,ax=None,figsize=(7.5,3)):
        if self.history:
            history = self.history
        else:
            raise RuntimeError("No history to plot -- run train() first.")

        # Get saved epoch
        results = self.model.evaluate(self.X_valid.load(), self.y_valid, verbose=0)
        
        saved_loss, saved_acc, saved_auc = results

        saved_epoch = [ i for i,v in enumerate(history.history['val_auc']) if round(v,3) == round(saved_auc,3) ][0]

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=figsize)
            
        ax.plot(history.history['val_auc'], label='val AUC', marker='.')
        ax.plot(history.history['auc'], label='train AUC', marker='.')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.axvline(saved_epoch, linestyle='--', c='r', label='Saved Epoch')
        all_auc = history.history['auc'] + history.history['val_auc']
        all_auc = np.array(all_auc)
        midxax = all_auc.min() + ((all_auc.max() - all_auc.min()) / 2)
        ax.text(saved_epoch+.2, midxax, f"val:    {saved_auc:.4f}\ntrain:  {history.history['auc'][saved_epoch]:.4f}")
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
        
        y_pred = self.model.predict(X.load(), verbose=0)

        roc_auc = roc_auc_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred)
        
        # if self.history is None:
        #     raise RuntimeError("No history -- run train() first.")
        # if self.y_prob is None:
        #     self.y_prob = self.model.predict(self.X_test.load(), verbose=0)
        # if self.threshold is None:
        #     self.threshold = self.optimalThreshold()
        # if self.y_pred is None:
        #     self.y_pred = np.where(self.y_prob > self.threshold, 1, 0)

        self.roc_auc = roc_auc

        if not ax:
            fig,ax = plt.subplots(1,1, figsize=figsize)

        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % self.roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic Curve')
        ax.legend(loc="lower right")

    def summary_plots(self):
        self.plotHistory()
        fig,(l,r)=plt.subplots(1,2,figsize=(8,3))
        self.plotCM(ax=l)
        self.plotROC(ax=r)

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
        
        self.summary_plots()



    