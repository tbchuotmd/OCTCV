"""
octcv.mdl_lib.architectures - CNN model architectures for OCT volume classification.

Provides:
    - buildSequential: Original 5-block Conv3D sequential model
    - buildResNet: ResNet-like model with skip connections
    - buildAttnNN: Attention-augmented ResNet (SE + spatial attention)
"""

import os
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import (Conv3D, Dense, BatchNormalization,
                          GlobalAveragePooling3D, ReLU, Input, Add)
from keras import Model as kerasModel
from keras import ops as Kops

# MODEL 1 : ORIGINAL SEQUENTIAL
def buildSequential(input_shape=(64, 128, 64, 1),
                    activation='sigmoid',
                    n_classes=1,
                    model_name='original'):
    """
    Build the original 5-block Conv3D sequential model.

    Architecture: 5× (Conv3D → BatchNorm → ReLU) → GAP3D → Dense(sigmoid).

    Parameters
    ----------
    input_shape : tuple
        Shape of input volumes (D, H, W, C). Default: (64, 128, 64, 1).
    activation : str
        Output activation. 'sigmoid' for binary, 'softmax' for multi-class.
    n_classes : int
        Number of output units. Forced to 1 when activation='sigmoid'.
    model_name : str
        Name prefix for the model.

    Returns
    -------
    keras.Sequential
    """
    
    if activation == 'sigmoid':
        n_classes = 1

    def conv3D_block(n_channels, kernel_size,strides=1, padding='same', layer_name=None):
        if layer_name: 
            cname = f"BLOCK{layer_name}_Conv3D"
            bname = f"BLOCK{layer_name}_BatchNorm"
            rname = f"BLOCK{layer_name}_ReLU"
        else:
            cname,bname,rname = None,None,None
            
        layer_list = [
            Conv3D(filters = n_channels,
                   kernel_size = kernel_size,
                   strides = strides,
                   padding = padding,
                   name = cname),
            BatchNormalization(name=bname),
            ReLU(name=rname)
        ]

        return layer_list

    kernelSizeSeq = [ 7,5,5,3,3 ]
    strideSeq = [ 2,1,1,1,1 ]
    
    five_Conv3D_blocks = [ ]
    for i,(k,s) in enumerate(zip(kernelSizeSeq,strideSeq)):
        five_Conv3D_blocks.extend( conv3D_block(32,k,s,layer_name=str(i+1))  )
        
    # Instantiate sequential model with architecture matching that described in paper
    return Sequential([

        # Input Layer
        Input(shape=input_shape),

        # Five Main Blocks, each involving Conv3D,BatchNormalization,and ReLU 
        *five_Conv3D_blocks,

        # Global Average Pooling
        GlobalAveragePooling3D(name='GlobalAveragePooling3D'),
    
        # Dense → Softmax (for multi-class CAM support)
        Dense(n_classes, activation=activation, name='Dense')  # use 1 + sigmoid if binary
    ], name = f"{model_name}_sequential")


# MODEL 2: RESNET-LIKE

def buildResNet(input_shape=(64, 128, 64, 1),
                activation='sigmoid', n_classes=1,
                name='ResNetLike'):
    """
    Build a ResNet-like model with residual skip connections.

    Architecture: Conv3D stem → 3× residual blocks → GAP3D → Dense.

    Parameters
    ----------
    input_shape : tuple
        Input volume shape. Default: (64, 128, 64, 1).
    activation : str
        Output activation function.
    n_classes : int
        Output units. Forced to 1 for sigmoid.
    name : str
        Model name.

    Returns
    -------
    keras.Model
    """
    
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
        
    def build_resnet_like_model(input_shape=input_shape,
                                activation=activation,
                                n_classes=n_classes,
                                name=name):
        
        if activation == 'sigmoid':
            n_classes = 1
        
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
        outputs = Dense(n_classes, activation=activation)(x)
        
        # Create and return the model
        return kerasModel(inputs, outputs,name=name)

    return build_resnet_like_model()

# MODEL 3: ATTENTION

def buildAttnNN(input_shape=(64, 128, 64, 1),
                activation='sigmoid', n_classes=1,
                name='Attention'):
    """
    Build an attention-augmented ResNet model with SE and spatial attention.

    Architecture: Conv3D stem → SE residual blocks → spatial attention → GAP3D → Dense.

    Parameters
    ----------
    input_shape : tuple
        Input volume shape. Default: (64, 128, 64, 1).
    activation : str
        Output activation function.
    n_classes : int
        Output units. Forced to 1 for sigmoid.
    name : str
        Model name.

    Returns
    -------
    keras.Model
    """
    
    if activation == 'sigmoid':
        n_classes = 1
        
    def se_block(x, r=16):
        c = x.shape[-1]
        s = layers.GlobalAveragePooling3D()(x)             # (B, C)
        s = layers.Dense(max(c // r, 4), activation='relu')(s)
        s = layers.Dense(c, activation='sigmoid')(s)
        s = layers.Reshape((1, 1, 1, c))(s)
        return layers.Multiply()([x, s])
    
   
    def spatial_attention(x, k=7):
        """Lightweight spatial attention for 3D (CBAM-style with avg+max)."""
        # channel-wise avg & max → concat → 3D conv → sigmoid mask
        avg = Kops.mean(x, axis=-1, keepdims=True)
        mx  = Kops.max(x,  axis=-1, keepdims=True)
        m   = layers.Concatenate(axis=-1)([avg, mx])           # (B,D,H,W,2)
        m   = layers.Conv3D(1, kernel_size=k, padding='same', activation='sigmoid')(m)
        return layers.Multiply()([x, m])
    
    def residual_block_attn(x, filters, kernel_size=3, strides=1,
                      use_se=True, se_r=16,
                      use_spatial=False, spatial_kernel=7):
        
        shortcut = x
    
        # Residual branch
        y = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
    
        y = Conv3D(filters, kernel_size=kernel_size, padding='same')(y)
        y = BatchNormalization()(y)
    
        # Optional attention on residual branch *before* addition
        if use_se:
            y = se_block(y, r=se_r)
        if use_spatial:
            y = spatial_attention(y, k=spatial_kernel)
    
        # Skip path if shape changes
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
    
        out = Add()([y, shortcut])
        out = ReLU()(out)
        return out

    def build_resnet_like_model_attn(input_shape=input_shape, 
                                     activation=activation,
                                     n_classes=n_classes,
                                     use_spatial_last=True,
                                     name=name):
        
        inputs = Input(shape=input_shape)
    
        x = Conv3D(16, kernel_size=5, strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        # Stage 1 (larger maps): SE only
        x = residual_block_attn(x, filters=16, strides=1, use_se=True, use_spatial=False)
    
        # Stage 2: downsample; SE only
        x = residual_block_attn(x, filters=32, strides=2, use_se=True, use_spatial=False)
    
        #x = residual_block_attn(x, filters=32, strides=2, use_se=True, use_spatial=False)
    
        # Stage 3 (smaller maps): SE + optional spatial
        x = residual_block_attn(x, filters=32, strides=1, use_se=True,
                              use_spatial=use_spatial_last, spatial_kernel=7)
    
        x = GlobalAveragePooling3D()(x)
    
        outputs = Dense(n_classes, activation=activation)(x)
    
        return kerasModel(inputs, outputs, name=name)

    return build_resnet_like_model_attn()
    