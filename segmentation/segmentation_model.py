'''
Created on 27-June-2021

@author: owaish
tensorflow >2.0
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Convolution2D,GlobalAveragePooling2D, Input,Multiply,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate,Reshape,Permute,Activation,UpSampling2D,ZeroPadding2D,Lambda, Dense,DepthwiseConv2D,Concatenate

#from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.keras.layers import AveragePooling2D, MaxPool2D
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras import constraints
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import LeakyReLU,PReLU,ReLU
from tensorflow.keras.layers import  Add
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

#from tensorflow.keras.utils.layer_utils import get_source_inputs
#from tensorflow.keras.utils.data_utils import get_file
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
#


# import tensorflow as tf
# import numpy as np
# from keras import layers
#
# from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Convolution2D,GlobalAveragePooling2D, Input,Multiply
# from keras.layers import Conv2DTranspose,concatenate,Reshape,Permute,Activation,UpSampling2D,ZeroPadding2D,Lambda, Dense,DepthwiseConv2D,Concatenate
#
# from keras.initializers import glorot_uniform
# from keras.layers.pooling import AveragePooling2D, MaxPool2D
# from keras import initializers
# from keras.regularizers import l2
# from keras import constraints
# from keras.models import Model, Sequential
#
# from keras.layers import LeakyReLU,PReLU,ReLU
# from keras.layers.merge import  Add
# import keras.backend as K
# import keras




# Part of code taken for  DeepLabV3 plus below author 
WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"
#============================================================================================
weight_decay = 0.0005

TOP_DOWN_PYRAMID_SIZE = 256



def grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = Concatenate()(group_list)
    x = BatchNormalization(axis=3)(group_merge)
    x = Activation('relu')(x)
    return x

def attention_gate(X, g, channel,  
                   activation='ReLU', 
                   attention='add', name='att'):
    '''
    Self-attention gate modified from Oktay et al. 2018.
    
    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')
    
    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X_att: output tensor.
    
    '''
    activation_func = eval(activation)
    attention_func = eval(attention)
    
    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)
    
    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)
    
    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    
    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)
    
    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #
    
    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    
    # multiplicative attention masking
    X_att = Multiply([X, coef_att], name='{}_masking'.format(name))
    
    return X_att



def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input
    grouped_channels = int(filters / cardinality)

    if init.shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([init, x])
    x = Activation('relu')(x)
    return x


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3]//2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

def sse_block(prevlayer, prefix):
    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv

def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', dilation_rate=(1,1),activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding,dilation_rate=dilation_rate,kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def Attention(x, prefix, attention_type = None):
    if  attention_type =='channel':
        return cse_block(x, prefix)
    elif attention_type =='spatial':
        return sse_block(x, prefix)
    elif attention_type =='channel_spatial':
        return csse_block(x, prefix)
    else:
        return x
               
    
def eassp(x, channel =256 , OS=16):
    #branching for Atrous Spatial Pyramid Pooling
    # Image Feature branch
    if OS == 8:
        atrous_rates = (3, 6, 9)
    else:
        atrous_rates = (6, 12, 18)

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(channel, (1, 1), padding='same',use_bias=False)(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
      
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    # print(size_before)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=False)(x)
    b0 = BatchNormalization( epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    
    
    
    b1 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)
    b1 = convolution_block(b1, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)
    b1 = convolution_block(b1, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[0],atrous_rates[0]),activation=True)   
    b1 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b1)
    b1 = BatchNormalization(epsilon=1e-5)(b1)
    b1 = Activation('relu')(b1)
        
    # rate = 12 (24)
    b2 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)
    b2 = convolution_block(b2, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)
    b2 = convolution_block(b2, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[1],atrous_rates[1]),activation=True)   
    b2 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b2)
    b2 = BatchNormalization(epsilon=1e-5)(b2)
    b2 = Activation('relu')(b2)
        
    # rate = 18 (36)
    b3 = convolution_block(x, channel//4, (1,1), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)
    b3 = convolution_block(b3, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)
    b3 = convolution_block(b3, channel//4, (3,3), strides=(1,1), padding='same', dilation_rate =(atrous_rates[2],atrous_rates[2]),activation=True)   
    b3 = Conv2D(channel,(1, 1), padding='same', use_bias=False)(b3)
    b3 = BatchNormalization(epsilon=1e-5)(b3)
    b3 = Activation('relu')(b3)
           
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(channel, (1, 1), padding='same',use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding,kernel_regularizer=l2(weight_decay), use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', kernel_regularizer=l2(weight_decay), use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', 
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
                      
                      
def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]#.value  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
              OS=16, alpha=1., activation=None):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

#     if not (weights in {'pascal_voc', 'cityscapes', None}):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `pascal_voc`, or `cityscapes` '
#                          '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name = 'aspp0_activation')(b0)

    #there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        skip_size = tf.keras.backend.int_shape(skip1)
        x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        skip_size[1:3],
                                                        method='bilinear', align_corners=True))(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)
    # Modififcation by me                                                
    
    reshape = Reshape((input_shape[0]*input_shape[1],classes), input_shape = input_shape)(x)
    x = Permute((1,2))(reshape)
    #softmax = Activation('softmax')(permut)
    #model = Model(inputs= [input_img], outputs= [softmax])


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation, dtype = 'float32')(x)

    model = Model(inputs, x, name='deeplabv3plus')

    # load weights

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    elif weights == 'cityscapes':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_MOBILE_CS,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
#added by me
    else:
        model.load_weights(weights,by_name=True)  
#         for c,l in enumerate(model.layers):
#                 if c<356:
#                     l.trainable= False
    return model


# def Deeplabv3V2(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
#               OS=16, alpha=1., activation=None):
#     """ Instantiates the Deeplabv3+ architecture
#     Optionally loads weights pre-trained
#     on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
#     # Arguments
#         weights: one of 'pascal_voc' (pre-trained on pascal voc),
#             'cityscapes' (pre-trained on cityscape) or None (random initialization)
#         input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#             to use as image input for the model.
#         input_shape: shape of input image. format HxWxC
#             PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
#         classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
#             If number of classes not aligned with the weights used, last layer is initialized randomly
#         backbone: backbone to use. one of {'xception','mobilenetv2'}
#         activation: optional activation to add to the top of the network.
#             One of 'softmax', 'sigmoid' or None
#         OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
#             Used only for xception backbone.
#         alpha: controls the width of the MobileNetV2 network. This is known as the
#             width multiplier in the MobileNetV2 paper.
#                 - If `alpha` < 1.0, proportionally decreases the number
#                     of filters in each layer.
#                 - If `alpha` > 1.0, proportionally increases the number
#                     of filters in each layer.
#                 - If `alpha` = 1, default number of filters from the paper
#                     are used at each layer.
#             Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
#     # Returns
#         A Keras model instance.
#     # Raises
#         RuntimeError: If attempting to run this model with a
#             backend that does not support separable convolutions.
#         ValueError: in case of invalid argument for `weights` or `backbone`
#     """
#
# #     if not (weights in {'pascal_voc', 'cityscapes', None}):
# #         raise ValueError('The `weights` argument should be either '
# #                          '`None` (random initialization), `pascal_voc`, or `cityscapes` '
# #                          '(pre-trained on PASCAL VOC)')
#
#     if not (backbone in {'xception', 'mobilenetv2'}):
#         raise ValueError('The `backbone` argument should be either '
#                          '`xception`  or `mobilenetv2` ')
#
#     if input_tensor is None:
#         img_input = Input(shape=input_shape)
#     else:
#         img_input = input_tensor
#
#     if backbone == 'xception':
#         if OS == 8:
#             entry_block3_stride = 1
#             middle_block_rate = 2  # ! Not mentioned in paper, but required
#             exit_block_rates = (2, 4)
#             atrous_rates = (12, 24, 36)
#         else:
#             entry_block3_stride = 2
#             middle_block_rate = 1
#             exit_block_rates = (1, 2)
#             atrous_rates = (6, 12, 18)
#
#         x = Conv2D(32, (3, 3), strides=(2, 2),
#                    name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
#         x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
#         x = Activation(tf.nn.relu)(x)
#
#         x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
#         x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
#         x = Activation(tf.nn.relu)(x)
#
#         x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
#                             skip_connection_type='conv', stride=2,
#                             depth_activation=False)
#         x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
#                                    skip_connection_type='conv', stride=2,
#                                    depth_activation=False, return_skip=True)
#
#         x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
#                             skip_connection_type='conv', stride=entry_block3_stride,
#                             depth_activation=False)
#         for i in range(16):
#             x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
#                                 skip_connection_type='sum', stride=1, rate=middle_block_rate,
#                                 depth_activation=False)
#
#         x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
#                             skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
#                             depth_activation=False)
#         x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
#                             skip_connection_type='none', stride=1, rate=exit_block_rates[1],
#                             depth_activation=True)
#
#     else:
#         OS = 8
#         first_block_filters = _make_divisible(32 * alpha, 8)
#         x = Conv2D(first_block_filters,
#                    kernel_size=3,
#                    strides=(2, 2), padding='same',
#                    use_bias=False, name='Conv')(img_input)
#         x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
#         x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)
#
#         x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
#                                 expansion=1, block_id=0, skip_connection=False)
#
#         x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
#                                 expansion=6, block_id=1, skip_connection=False)
#         x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
#                                 expansion=6, block_id=2, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
#                                 expansion=6, block_id=3, skip_connection=False)
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                                 expansion=6, block_id=4, skip_connection=True)
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                                 expansion=6, block_id=5, skip_connection=True)
#
#         # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
#                                 expansion=6, block_id=6, skip_connection=False)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=7, skip_connection=True)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=8, skip_connection=True)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=9, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=10, skip_connection=False)
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=11, skip_connection=True)
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=12, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
#                                 expansion=6, block_id=13, skip_connection=False)
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=14, skip_connection=True)
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=15, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=16, skip_connection=False)
#
#     # end of feature extractor
#
#     # branching for Atrous Spatial Pyramid Pooling
#
#     # Image Feature branch
#     shape_before = tf.shape(x)
#     b4 = GlobalAveragePooling2D()(x)
#     # from (b_size, channels)->(b_size, 1, 1, channels)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Conv2D(256, (1, 1), padding='same',
#                 use_bias=False, name='image_pooling')(b4)
#     b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
#     b4 = Activation('relu')(b4)
#
#     # upsample. have to use compat because of the option align_corners
#     size_before = tf.keras.backend.int_shape(x)
#     b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
#                                                     method='bilinear', align_corners=True))(b4)
#     # simple 1x1
#     b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
#     b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
#     b0 = Activation('relu', name='aspp0_activation')(b0)
#     #there are only 2 branches in mobilenetV2. not sure why
#     if backbone == 'xception':
#         # rate = 6 (12)
#         b1 = SepConv_BN(x, 64, 'aspp1',
#                             rate=atrous_rates[0], kernel_size=1, depth_activation=True, epsilon=1e-5)
#         b1 = SepConv_BN(b1, 64, 'aspp1_1',
#                             rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#         b1 = SepConv_BN(b1, 64, 'aspp1_2',
#                              rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#     #          
#         b1 = Conv2D(256,(1, 1), padding='same', use_bias=False, name='easpp1_1')(b1)
#         b1 = BatchNormalization(name='easpp1_1_BN', epsilon=1e-5)(b1)
#         b1 = Activation('relu', name='easpp1_1_activation')(b1)
#
#             # rate = 12 (24)
#         b2 = SepConv_BN(x, 64, 'assp2',
#                             rate=atrous_rates[1], kernel_size=1,depth_activation=True, epsilon=1e-5)
#         b2 = SepConv_BN(b2, 64, 'assp2_1',
#                             rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#         b2 = SepConv_BN(b2, 64, 'assp2_2',
#                             rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#     #          
#         b2 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='eassp2_1')(b2)
#         b2 = BatchNormalization(name='eassp2_1_BN', epsilon=1e-5)(b2)
#         b2 = Activation('relu', name='eassp2_1_activation')(b2)
#
#             # rate = 18 (36)
#         b3 = SepConv_BN(x,  64, 'assp3',
#                             rate=atrous_rates[2],kernel_size=1, depth_activation=True, epsilon=1e-5)
#         b3 = SepConv_BN(b3, 64, 'assp3_1',
#                             rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
#         b3 = SepConv_BN(b3, 64, 'assp3_2',
#                             rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
#     #           
#         b3 = Conv2D(256, (1, 1), padding='same', use_bias=False,name='eassp3_1')(b3)
#         b3 = BatchNormalization(name='eassp3_1_BN', epsilon=1e-5)(b3)
#         b3 = Activation('relu', name='eassp3_1_activation')(b3)
#
#         # concatenate ASPP branches & project
#         x = Concatenate()([b4, b0, b1, b2, b3])
#
#     else:
#         x = Concatenate()([b4, b0])
#
#     x = Conv2D(256, (1, 1), padding='same',
#                use_bias=False, name='concat_projection')(x)
#     x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
#     x = Activation(tf.nn.relu)(x)
#     x = Dropout(0.1)(x)
#
#
#     # DeepLab v.3+ decoder
#
#     if backbone == 'xception':
#         # Feature projection
#         # x4 (x2) block
#         skip_size = tf.keras.backend.int_shape(skip1)
#         x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
#                                                             skip_size[1:3],
#                                                             method='bilinear', align_corners=True))(x)
#
#         dec_skip1 = Conv2D(48, (1, 1), padding='same',
#                                use_bias=False, name='feature_projection0')(skip1)
#         dec_skip1 = BatchNormalization(
#                 name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
#         dec_skip1 = Activation('relu')(dec_skip1)
#         x = Concatenate()([x, dec_skip1])
#         x = Dropout(0.1)(x)
#
#
#         x = SepConv_BN(x, 64, 'decoder_conv0',kernel_size=1,
#                            depth_activation=True, epsilon=1e-5)
#         x = SepConv_BN(x, 64, 'decoder_conv0_1',kernel_size=3,
#                            depth_activation=True, epsilon=1e-5)
#         x = SepConv_BN(x, 64, 'decoder_conv0_2',kernel_size=3,
#                            depth_activation=True, epsilon=1e-5)
#
#         x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv0_2_1')(x)
#         x = BatchNormalization(name='decoder_conv0_2_1_BN', epsilon=1e-5)(x)
#         x = Activation('relu', name='decoder_conv0_2_1_activation')(x)
#
#         x = SepConv_BN(x, 64, 'decoder_conv1',kernel_size=1,
#                            depth_activation=True, epsilon=1e-5)
#         x = SepConv_BN(x, 64, 'decoder_conv1_1',kernel_size=3,
#                            depth_activation=True, epsilon=1e-5)
#         x = SepConv_BN(x, 64, 'decoder_conv1_2',kernel_size=3,
#                            depth_activation=True, epsilon=1e-5)
#
#         x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv1_2_1')(x)
#         x = BatchNormalization(name='decoder_conv1_2_1_BN', epsilon=1e-5)(x)
#         x = Activation('relu', name='decoder_conv1_2_1_activation')(x)
#         x = Dropout(0.1)(x)
#
#     # you can use it with arbitary number of classes
#     if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
#         last_layer_name = 'logits_semantic'
#     else:
#         last_layer_name = 'custom_logits_semantic'
#
#     x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#     size_before3 = tf.keras.backend.int_shape(img_input)
#     x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
#                                                     size_before3[1:3],
#                                                     method='bilinear', align_corners=True))(x)
#     # Modififcation by me                                                
#
#     reshape = Reshape((input_shape[0]*input_shape[1],classes), input_shape = input_shape)(x)
#     x = Permute((1,2))(reshape)
#     #softmax = Activation('softmax')(permut)
#     #model = Model(inputs= [input_img], outputs= [softmax])
#
#
#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = get_source_inputs(input_tensor)
#     else:
#         inputs = img_input
#
#     if activation in {'softmax', 'sigmoid'}:
#         x = tf.keras.layers.Activation(activation, dtype = 'float32')(x)
#
#     model = Model(inputs, x, name='deeplabv3plus')
#
#     if OS == 8:
#         entry_block3_stride = 1
#         middle_block_rate = 2  # ! Not mentioned in paper, but required
#         exit_block_rates = (2, 4)
#         atrous_rates = (12, 24, 36)
#     else:
#         entry_block3_stride = 2
#         middle_block_rate = 1
#         exit_block_rates = (1, 2)
#         atrous_rates = (6, 12, 18)
#
#     filters = int(x.shape[-1])
#
#     x_bottom = UpSampling2D(size=(2, 2),interpolation="bilinear")(x_bottom)
#     x_bottom = Conv2D(filters, (1, 1), padding='same',kernel_initializer='he_normal', use_bias=False)(x_bottom)
#     x_bottom = BatchNormalization(name='matching_filter_BN', epsilon=1e-5)(x_bottom)
#     x_bottom = Activation('relu')(x_bottom)
#
#
#     #x_bottom = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before[1:3],
#     #                                                method='nearest', align_corners=True))(x_bottom)
#
#     #x = Concatenate()([x_bottom,x])
#     #x = Multiply()([x_bottom,x])
#     x = Add()([x_bottom,x]) 
#     x = Dropout(0.1)(x)
#
#
#     #branching for Atrous Spatial Pyramid Pooling
#
#     # Image Feature branch
#     shape_before = tf.shape(x)
#     b4 = GlobalAveragePooling2D()(x)
#     # from (b_size, channels)->(b_size, 1, 1, channels)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Conv2D(256, (1, 1), padding='same',
#                 use_bias=False, name='image_pooling')(b4)
#     b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
#     b4 = Activation('relu')(b4)
#
#     # upsample. have to use compat because of the option align_corners
#     size_before = tf.keras.backend.int_shape(x)
#     b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
#                                                     method='bilinear', align_corners=True))(b4)
#     # simple 1x1
#     b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
#     b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
#     b0 = Activation('relu', name='aspp0_activation')(b0)
#
#
#
#     b1 = SepConv_BN(x, 64, 'aspp1',
#                         rate=atrous_rates[0], kernel_size=1, depth_activation=True, epsilon=1e-5)
#     b1 = SepConv_BN(b1, 64, 'aspp1_1',
#                         rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#     b1 = SepConv_BN(b1, 64, 'aspp1_2',
#                          rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
# #          
#     b1 = Conv2D(256,(1, 1), padding='same', use_bias=False, name='easpp1_1')(b1)
#     b1 = BatchNormalization(name='easpp1_1_BN', epsilon=1e-5)(b1)
#     b1 = Activation('relu', name='easpp1_1_activation')(b1)
#
#         # rate = 12 (24)
#     b2 = SepConv_BN(x, 64, 'assp2',
#                         rate=atrous_rates[1], kernel_size=1,depth_activation=True, epsilon=1e-5)
#     b2 = SepConv_BN(b2, 64, 'assp2_1',
#
#     # load weights
#
# #     if weights == 'pascal_voc':
# #         if backbone == 'xception':
# #             weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
# #                                     WEIGHTS_PATH_X,
# #                                     cache_subdir='models')
# #         else:
# #             weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
# #                                     WEIGHTS_PATH_MOBILE,
# #                                     cache_subdir='models')
# #         model.load_weights(weights_path, by_name=True)
# #     elif weights == 'cityscapes':
# #         if backbone == 'xception':
# #             weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
# #                                     WEIGHTS_PATH_X_CS,
# #                                     cache_subdir='models')
# #         else:
# #             weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
# #                                     WEIGHTS_PATH_MOBILE_CS,
# #                                     cache_subdir='models')
# #         model.load_weights(weights_path, by_name=True)
# # #added by me
# #     else:
# #         model.load_weights(weights,by_name=True)  
# #         for c,l in enumerate(model.layers):
# #                 if c<356:
# #                     l.trainable= False
#     return model
 
def Deeplabv3Plus_resnet(weights=None, input_tensor=None, input_shape=(512, 512, 3), classes=2, backbone='renet50',
              OS=16, activation='softmax'):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
        
    encoder = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    input_im = encoder.input
    x = encoder.get_layer('conv4_block6_out').output
    #x_bottom = encoder.get_layer('conv5_block3_out').output
    skip1 =  encoder.get_layer('conv2_block3_out').output
    
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)
        
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name = 'aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name = 'aspp0_activation')(b0)
    #there are only 2 branches in mobilenetV2. not sure why
    
        # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    # DeepLab v.3+ decoder
        # Feature projection
        # x4 (x2) block
    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    skip_size[1:3],
                                                    method='bilinear', align_corners=True))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)



    x = Conv2D(classes, (1, 1), padding='same')(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)
    # Modififcation by me                                                
    
    reshape = Reshape((input_shape[0]*input_shape[1],classes), input_shape = input_shape)(x)
    x = Permute((1,2))(reshape)
    #softmax = Activation('softmax')(permut)
    #model = Model(inputs= [input_img], outputs= [softmax])


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation, dtype = 'float32')(x)
    model = Model(inputs=[input_im],outputs=[x], name='deeplabv3plus')
    if weights is not None:
        model.load_weights(weights)
    return model

def Deeplabv3_eassp(weights=None, input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='efficientnetb0',
              OS=16, alpha=1., activation='softmax'):
    """ Instantiates the Deeplabv3+ architecture
    # Arguments
        weights: one of 'noisy-student' (pre-trained on noisy-student),
            'imagenet' (pre-trained on imagenet) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            
        classes: number of desired classes. .
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4',
                         'efficientnetb5','efficientnetb6','efficientnetb7'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

#     if not (weights in {'pascal_voc', 'cityscapes', None}):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `pascal_voc`, or `cityscapes` '
#                          '(pre-trained on PASCAL VOC)')
    
    if not (backbone in {'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3','efficientnetb4',
                         'efficientnetb5','efficientnetb6','efficientnetb7'}):
        raise ValueError('The `backbone` argument should be in '
                         '`efficientnet`  class ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
        
 #   import efficientnet.keras as efn
    import efficientnet.tfkeras as efn
    
    if backbone == 'efficientnetb0': 
        encoder = efn.EfficientNetB0(input_shape=input_shape,include_top=False, weights='noisy-student')#'noisy-student')
        
    elif backbone == 'efficientnetb1':
        encoder = efn.EfficientNetB1(input_shape=input_shape,include_top=False, weights='noisy-student')#'imagenet')#
        
    elif backbone == 'efficientnetb2':
        encoder = efn.EfficientNetB2(input_shape=input_shape,include_top=False, weights='noisy-student')#'imagenet')
        
    elif backbone == 'efficientnetb3':
        encoder = efn.EfficientNetB3(input_shape=input_shape,include_top=False, weights='noisy-student')
       
    elif backbone == 'efficientnetb4':
        encoder = efn.EfficientNetB4(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb5':
        encoder = efn.EfficientNetB5(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb6':
        encoder = efn.EfficientNetB6(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    elif backbone == 'efficientnetb7':
        encoder = efn.EfficientNetB7(input_shape=input_shape,include_top=False, weights='noisy-student')
        
    #encoder.summary()
    input_im = encoder.input
    x = encoder.get_layer('block6a_expand_activation').output
    #x_bottom = encoder.get_layer('top_activation').output
    skip1 =  encoder.get_layer('block3a_expand_activation').output
    #skip2 =  encoder.get_layer('block3a_expand_activation').output
    
    
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)
    
    #filters = int(x.shape[-1])
    
    #x_bottom = UpSampling2D(size=(2, 2),interpolation="bilinear")(x_bottom)
    #x_bottom = Conv2D(filters, (1, 1), padding='same',kernel_initializer='he_normal', use_bias=False)(x_bottom)
    #x_bottom = BatchNormalization(name='matching_filter_BN', epsilon=1e-5)(x_bottom)
    #x_bottom = Activation('relu')(x_bottom)
    
    
    #x_bottom = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before[1:3],
    #                                                method='nearest', align_corners=True))(x_bottom)
                                                  
    #x = Concatenate()([x_bottom,x])
    #x = Multiply()([x_bottom,x])
    #x = Add()([x_bottom,x]) 
    #x = Dropout(0.1)(x)

      
    #branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
      
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    
    
    
    b1 = SepConv_BN(x, 64, 'aspp1',
                        rate=atrous_rates[0], kernel_size=1, depth_activation=True, epsilon=1e-5)
    b1 = SepConv_BN(b1, 64, 'aspp1_1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b1 = SepConv_BN(b1, 64, 'aspp1_2',
                         rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#          
    b1 = Conv2D(256,(1, 1), padding='same', use_bias=False, name='easpp1_1')(b1)
    b1 = BatchNormalization(name='easpp1_1_BN', epsilon=1e-5)(b1)
    b1 = Activation('relu', name='easpp1_1_activation')(b1)
        
        # rate = 12 (24)
    b2 = SepConv_BN(x, 64, 'assp2',
                        rate=atrous_rates[1], kernel_size=1,depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(b2, 64, 'assp2_1',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(b2, 64, 'assp2_2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#          
    b2 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='eassp2_1')(b2)
    b2 = BatchNormalization(name='eassp2_1_BN', epsilon=1e-5)(b2)
    b2 = Activation('relu', name='eassp2_1_activation')(b2)
        
        # rate = 18 (36)
    b3 = SepConv_BN(x,  64, 'assp3',
                        rate=atrous_rates[2],kernel_size=1, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(b3, 64, 'assp3_1',
                        rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(b3, 64, 'assp3_2',
                        rate=atrous_rates[2],depth_activation=True, epsilon=1e-5)
#           
    b3 = Conv2D(256, (1, 1), padding='same', use_bias=False,name='eassp3_1')(b3)
    b3 = BatchNormalization(name='eassp3_1_BN', epsilon=1e-5)(b3)
    b3 = Activation('relu', name='eassp3_1_activation')(b3)

            
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Dropout(0.1)(x)
    
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
   
    


# Decoder
    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        skip_size[1:3],
                                                        method='bilinear', align_corners=True))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = Dropout(0.1)(x)


    x = SepConv_BN(x, 64, 'decoder_conv0',kernel_size=1,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv0_1',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv0_2',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv0_2_1')(x)
    x = BatchNormalization(name='decoder_conv0_2_1_BN', epsilon=1e-5)(x)
    x = Activation('relu', name='decoder_conv0_2_1_activation')(x)
    
    x = SepConv_BN(x, 64, 'decoder_conv1',kernel_size=1,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv1_1',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 64, 'decoder_conv1_2',kernel_size=3,
                       depth_activation=True, epsilon=1e-5)
    
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='decoder_conv1_2_1')(x)
    x = BatchNormalization(name='decoder_conv1_2_1_BN', epsilon=1e-5)(x)
    x = Activation('relu', name='decoder_conv1_2_1_activation')(x)
    x = Dropout(0.1)(x)

    
    x = Conv2D(classes, (1, 1), padding='same', name='last_layer_befor_acivation')(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)
    # Modififcation by me                                                
    
    reshape = Reshape((input_shape[0]*input_shape[1],classes), input_shape = input_shape)(x)
    x = Permute((1,2))(reshape)
    
    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation,dtype='float32')(x)

    model = Model(inputs=[input_im],outputs=[x], name='deeplabv3plus')
    if weights is not None:
        model.load_weights(weights)
        # for c, l in enumerate(model.layers):
        #     if c<507:
        #         l.trainable= False
        #     else:
        #         l.trainable= True
    return model

def UEfficientNet_customize(input_shape=(None, None, 3),classes = 4,backbone = 'efficientnetb3',e_assp =False,dropout_rate=0.1,weightsPath=None):
    
    data_shape = input_shape[0]*input_shape[1]
    input_img = (input_shape[0], input_shape[1], classes)
    out_shape=(data_shape, classes)
    #import efficientnet.keras as efn
    import efficientnet.tfkeras as efn
    if backbone == 'efficientnetb0':
        backbone = efn.EfficientNetB0(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
    elif backbone == 'efficientnetb1':
        backbone = efn.EfficientNetB1(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
    elif backbone == 'efficientnetb2':
        backbone = efn.EfficientNetB2(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)    
        
    elif backbone == 'efficientnetb3':
        backbone = efn.EfficientNetB3(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
    elif backbone == 'efficientnetb4':
        backbone = efn.EfficientNetB4(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
    elif backbone == 'efficientnetb5':
        backbone = efn.EfficientNetB5(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
    
    elif backbone == 'efficientnetb6':
        backbone = efn.EfficientNetB6(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
    elif backbone == 'efficientnetb6':
        backbone = efn.EfficientNetB7(weights='noisy-student',
                            include_top=False,
                            input_shape=input_shape)
        
        
    else:
        print('invalid backbone')
        
        
    input = backbone.input
    start_neurons = 8 

    conv4 = backbone.get_layer('block6a_expand_activation').output#735
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    
    #conv4 = eassp(conv4,OS=16)
    
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    
    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle')(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    
    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)

    
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    deconv4_up4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up3)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4) 
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4) 
    
    
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.get_layer('block4a_expand_activation').output#379
    uconv3 = concatenate([deconv3,deconv4_up1, conv3])    
    uconv3 = Dropout(dropout_rate)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    conv2 = backbone.get_layer('block3a_expand_activation').output#246
    uconv2 = concatenate([deconv2,deconv3_up1,deconv4_up2, conv2])
    
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    
     
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.get_layer('block2a_expand_activation').output#143
    
    uconv1 = concatenate([deconv1,deconv2_up1,deconv3_up2,deconv4_up3, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    #if e_assp:
    # #    deconv4 = eassp(uconv0,filters = 32,OS=16)
    #
    # output_layer = Conv2D(classes, (1,1), padding="same")(uconv0)  
    # reshape = Reshape(out_shape, input_shape = input_img)(output_layer)  
    # softmax = Activation('softmax',dtype='float32')(reshape)
    
    output_layer = Conv2D(classes, (1,1), padding="same")(uconv0) 
    #output_layer = UpSampling2D(size=(4, 4),interpolation="bilinear")(output_layer)
    reshape = Reshape(out_shape, input_shape = input_img)(output_layer)  
    softmax1 = Activation('softmax',dtype='float32')(reshape)

    model = Model([input],[softmax1])
    
    if weightsPath is not None:
        model.load_weights(weightsPath) 
    return model

def resnext_fpn(input_shape, nb_labels, weights= None,depth=(3, 4, 6, 3),unet = False, e_assp = False, attention = None,
                cardinality=32, width=4, pooling = None,weight_decay=5e-4, batch_norm=True, batch_momentum=0.9):
    """
    TODO: add dilated convolutions as well
    Resnext-50 is defined by (3, 4, 6, 3) [default]
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    :param input_shape:
    :param nb_labels:
    :param depth:
    :param cardinality:
    :param width:
    :param weight_decay:
    :param batch_norm:
    :param batch_momentum:
    :return:
    """
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)
    
    bn_axis = 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)   #1/2
    #print(x.shape, 'X')
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Attention(x, prefix='stage_1',attention_type = attention)
    stage_1 = x                                                                         #(1/2^2)
    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x
    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay) #(1/2^3)
    x = Attention(x, prefix='stage_2',attention_type = attention)
    
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x
    x = bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)       #(1/2^4)
    x = Attention(x, prefix='stage_3',attention_type = attention)
    
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x
    #print(stage_4.shape, 'stage_4')
    x = bottleneck_block(x, 1024, cardinality, strides=2, weight_decay=weight_decay)     #(1/2^5)
    x = Attention(x, prefix='stage_4',attention_type = attention) 
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, weight_decay=weight_decay)  
    x = Attention(x, prefix='stage_5',attention_type = attention)
    
    if pooling is not None:
        size_before = tf.keras.backend.int_shape(x)
        if pooling == 'avg': 
            x = GlobalAveragePooling2D()(x)
            x = Lambda(lambda x: K.expand_dims(x, 1))(x)
            x = Lambda(lambda x: K.expand_dims(x, 1))(x)
            x = Conv2D(256, (1, 1), padding='same',use_bias=False)(x)
            x = BatchNormalization(epsilon=1e-5)(x)
            x = Activation('relu')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
            x = Lambda(lambda x: K.expand_dims(x, 1))(x)
            x = Lambda(lambda x: K.expand_dims(x, 1))(x)
            x = Conv2D(256, (1, 1), padding='same',use_bias=False)(x)
            x = BatchNormalization(epsilon=1e-5)(x)
            x = Activation('relu')(x)
        x = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(x)
    stage_5 = x
    #print(stage_5.shape, 'stage_5')
        
        
        
    
    # P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(stage_5)
    # P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
    #                             Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4', padding='same')(stage_4)])
    # P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
    #                             Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(stage_3)])
    # P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
    #                             Conv2D(TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
    # # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    # P2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="fpn_p2",kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(P2)
    # P3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="fpn_p3",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P3)
    # P4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="fpn_p4",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P4)
    # P5 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 2, (3, 3), padding="SAME", name="fpn_p5",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P5)
    #
    # head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head1_conv",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P2)
    # head1 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head1_conv_2",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(head1)
    # head1 = Dropout(0.1)(head1)
    #
    # head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head2_conv",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P3)
    # head2 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head2_conv_2",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(head2)
    # head2 = Dropout(0.1)(head2)
    #
    # head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head3_conv",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P4)
    # head3 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head3_conv_2",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(head3)
    # head3 = Dropout(0.1)(head3)
    #
    # head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head4_conv",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(P5)
    # head4 = Conv2D(TOP_DOWN_PYRAMID_SIZE // 4, (3, 3), padding="SAME", name="head4_conv_2",kernel_initializer='he_normal',kernel_regularizer=l2(weight_decay))(head4)
    # head4 = Dropout(0.1)(head4)
    #
    # f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
    # f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
    # f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
    # f_p5 = head1
    #
    # x = Concatenate()([f_p2, f_p3, f_p4, f_p5])
    #
    # #x = eassp(x, channel=64,  OS = 16)
    
    if unet:
        start_neurons = 8
        stage_5 = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle',kernel_initializer='he_normal')(stage_5)
        stage_5 = residual_block(stage_5,start_neurons * 32)
        #stage_5 = residual_block(stage_5,start_neurons * 32)
        stage_5 = LeakyReLU(alpha=0.1)(stage_5)
        
                
                
        stage_4_up = Conv2DTranspose(start_neurons * 16, (3, 3),  strides=(2, 2), padding="same")(stage_5)  #64,64
        stage_4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(stage_4_up)
        stage_4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(stage_4_up1)
        stage_4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(stage_4_up2)
        stage_4 = Concatenate()([stage_4_up, stage_4])
        stage_4 = Conv2D(start_neurons * 16, (1, 1), activation=None, padding="same",name='satge_4', kernel_initializer='he_normal')(stage_4)
        # stage_4 = residual_block(stage_4,start_neurons * 16)
        # stage_4 = LeakyReLU(alpha=0.1)(stage_4)
        stage_4 = Dropout(0.1)(stage_4) 
        
        
        stage_3_up = Conv2DTranspose(start_neurons * 8, (3, 3),  strides=(2, 2), padding="same")(stage_4)  #128,128
        stage_3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(stage_3_up)
        stage_3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(stage_3_up1)
        stage_3 = Concatenate()([stage_3_up,stage_4_up1, stage_3])
        stage_3 = Conv2D(start_neurons * 8, (1, 1), activation=None, padding="same",name='stage_3')(stage_3)
        # stage_3 = residual_block(stage_3,start_neurons * 8)
        # stage_3 = LeakyReLU(alpha=0.1)(stage_3)
        stage_3 = Dropout(0.1)(stage_3)
        
        
        stage_2_up = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(stage_3)  #256,256
        #stage_2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(stage_2_up) #512
        stage_2 = Concatenate()([stage_2_up,stage_3_up1,stage_4_up2,stage_2])
        stage_2 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same",name='stage_2')(stage_2)
        # stage_2 = residual_block(stage_2,start_neurons * 4)
        # stage_2 = LeakyReLU(alpha=0.1)(stage_2)
        stage_2 = Dropout(0.1)(stage_2)
        
        
        stage_1_up = Conv2D(start_neurons * 2, (3, 3), strides=(1, 1), padding="same")(stage_2) #256,256
        stage_1 = Concatenate()([stage_2_up,stage_3_up1,stage_4_up2, stage_1])
        stage_1 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same",name='stage_1')(stage_1)
        # stage_1 = residual_block(stage_1,start_neurons * 2)
        # stage_1 = LeakyReLU(alpha=0.1)(stage_1)
        #x = Concatenate()([x,stage_1])
        x = Dropout(0.1)(stage_1)
        
    
    #x = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv", kernel_initializer='he_normal',activation='linear')(x)
    x = Conv2D(nb_labels, (1, 1), padding="SAME", name="final_conv", kernel_initializer='he_normal')(x)
    x = UpSampling2D(size=(4, 4), name="final_upsample")(x)
    x = Reshape((input_shape[0]*input_shape[1],nb_labels), input_shape = input_shape)(x)
    x = Activation('softmax',dtype='float32')(x)
    
    model = Model(input_tensor, x)
    if weights is not None:
        model.load_weights(weights) 
    return model

def Models(width,height,depth,classes, model_type= 'unet_3plus_2d',backbone='ResNet50V2'):
    
    '''Unet++(model_type ='unet_plus_2d') ,ResUnet-a(model_type ='resunet_a_2d'), U^2-Net(model_type ='u2net_2d), Attention U-net(model_type = 'att_unet_2d'),
    '''
    name = model_type
    activation = 'ReLU'
    filter_num_down = [32, 64, 128, 256,512]
    filter_num_skip = [32, 32, 32, 32]
    filter_num_aggregate = 160
    weight=width
    height=height
    depth=3
    stack_num_down = 2  
    stack_num_up = 2  #earlier it was 1 for MPUH model
    n_labels = classes
    #atten_activation='ReLU', attention='add', output_activation='Softmax'
    from keras_unet_collection import models
    from keras_unet_collection import losses
    #loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    # unet3plus.compile(loss=[hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss, hybrid_loss],
    #               loss_weights=[0.25, 0.25, 0.25, 0.25, 1.0],
    #               optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    #loss_iou = losses.iou_seg(y_true, y_pred)
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    #(x) 
    # Classification-guided Module (CGM)
    #X_CGM = Dropout(rate=0.1)(X_CGM)
    #X_CGM = Conv2D(filter_num_skip[-1], 1, padding='same')(X_CGM)
    #X_CGM = GlobalMaxPooling2D()(X_CGM)
    #X_CGM = Activation('sigmoid',dtype = np.uint8)(X_CGM)  # for non-organ (binary) classification
    
    #X_CGM = Activation('softmax',dtype = np.uint8) (X_CGM) # for multiclass classification
    #from tensorflow.keras.backend import max
    #CGM_mask = max(X_CGM, axis= -1)
    
    # freezing batchnorm
    # from tensorflow.keras.layers import BatchNormalization    
    # for layer in model.layers:
    #     if isinstance(layer, BatchNormalization):
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False
    
    if model_type== 'unet_3plus_2d':
    
        base_model =  models.unet_3plus_2d([width,height,depth],n_labels = n_labels,
                                       stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                                       freeze_backbone = False, 
                                       freeze_batch_norm = False,
                                       filter_num_down = filter_num_down,
                                       backbone=backbone,
                                       #activation=activation, 
                                       #deep_supervision=True
                                       #batch_norm=True, pool=True, unpool=True, backbone=None, name=name
                                       )
    elif model_type == 'transunet_2d':
        base_model = models.u2net_2d([width,height,depth],n_labels = n_labels,
                                       filter_num_down =filter_num_down,
                                       #freeze_backbone=False, 
                                       #freeze_batch_norm=False,
                                       #backbone=backbone
                                       )
    elif model_type == 'unet_plus_2d':   #Unet++
        base_model = models.unet_plus_2d([width,height,depth],n_labels = n_labels,
                            filter_num =filter_num_down,
                            freeze_backbone=False, 
                            freeze_batch_norm=False,
                            backbone=backbone
                                       )
    elif model_type =='resunet_a_2d':  #ResUnet-a
        base_model =  models.resunet_a_2d([width,height,depth],n_labels = n_labels,
                            #stack_num_down=stack_num_down, 
                            #stack_num_up=stack_num_up,
                            #freeze_backbone=False, 
                            #freeze_batch_norm=False,
                            filter_num = filter_num_down,
                            #backbone=backbone
                            )

    
    elif model_type == 'u2net_2d':     #U^2-Net
        base_model =  models.u2net_2d([width,height,depth],n_labels = n_labels,
                            #stack_num_down=stack_num_down, 
                            #stack_num_up=stack_num_up,
                            #freeze_backbone=False,
                            #freeze_batch_norm=False,
                            filter_num_down = filter_num_down,
                            #backbone=backbone
                            )
    
    elif model_type == 'att_unet_2d':  #Attention U-net
        base_model =  models.att_unet_2d([width,height,depth],n_labels = n_labels,
                            stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                            freeze_backbone=False, freeze_batch_norm=False,
                            filter_num = filter_num_down,
                            backbone=backbone
                            )

        
        
    input_im = base_model.input
    output = base_model.output
    input_shape = (weight,height,classes)
    out_shape = (weight*height,classes)
    #input_img1 = tf.keras.Input(shape = (weight,height,depth))
    output = tf.keras.layers.Reshape(out_shape,input_shape = input_shape)(output)
    model = tf.keras.Model(input_im,output)
    return model



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # # # model= model =  Models(1024,1024,3,5,model_type = 'unet_plus_2d',backbone= 'EfficientNetB3')
# # # #model = resnext_fpn(input_shape=(1024,1024,3), nb_labels=5, unet = True, e_assp = False,attention=None)#'channel_spatial')#'spatial','channel_spatial'
# # model = Deeplabv3(input_shape=(512, 512, 3), classes=21, backbone='xception',
# #              OS=16, alpha=1., activation=None)
# model = Deeplabv3_eassp(weights =None,input_shape=(512, 512, 3), classes=2, backbone='efficientnetb0',OS=16, activation='softmax')
# #model = Deeplabv3Plus_resnet(input_shape=(512, 512, 3), classes=2, backbone='renet50',OS=16, activation='softmax')
# print(model.summary())
# print(len(model.layers))





