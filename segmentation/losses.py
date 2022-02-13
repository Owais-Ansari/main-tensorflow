'''
Created on 29-Aug-2019

@author: Owaish
'''
"""
Define our custom loss function.
"""
#from keras import backend as K
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
#import dill


def weighted_categorical_crossentropy_dice(y_true,y_pred,classes=2):
    
    'calculate the inverse weights'
    smooth=1.e-5          
    loss = 0
    gdc = 0 
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    total =  tf.reduce_sum(y_true[:,:,:])
    for i in range(classes):
        
        neuminator = tf.reduce_sum(y_true[:,:,i]*y_pred[:,:,i])
        denominator = tf.reduce_sum(y_true[:,:,i]+y_pred[:,:,i])
        gdc+= 2.*(neuminator+smooth)/(denominator+smooth)
        dice_loss = 1-gdc/classes  
        ce_loss =  -y_true[:,:,i] * tf.math.log(y_pred[:,:,i])   
        w = tf.reduce_sum(y_true[:,:,i])
        loss +=  (ce_loss+dice_loss)*(total - w)/total  
        #loss +=  (ce_loss)*(total - w)/total 
    return loss


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = tf.keras.backend.epsilon()  #K.epsilon()
        # clip to prevent NaN's and Inf's
     
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)

        return -tf.reduce_sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               -tf.reduce_sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.reduce_sum(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return tf.reduce_sum(loss, axis=1)
    scale_factor =10
    return categorical_focal_loss_fixed * scale_factor

smooth = 1


# useful for binary class only
def dice_score(epsilon=1):
    # with epsilon
    def dice_scr(y_true, y_pred):
        return 1-((2. * K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon()))
    return dice_scr


def inv_weight_multi_class_Dice_loss(y_true,y_pred,classes=6):
    smooth=1.e-5          
    loss = 0
    gdc = 0 
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    total =  tf.reduce_sum(y_true[:,:,:])
    for i in range(classes):
        neuminator = tf.reduce_sum(y_true[:,:,i]*y_pred[:,:,i])
        denominator = tf.reduce_sum(y_true[:,:,i]+y_pred[:,:,i])
        gdc+= 2.*(neuminator+smooth)/(denominator+smooth)
        dice_loss = 1-gdc/classes  
        #ce_loss =  -y_true[:,:,i] * tf.math.log(y_pred[:,:,i])   
        w = tf.reduce_sum(y_true[:,:,i])
        #loss +=  (ce_loss+dice_loss)*(total - w)/total
        loss +=  (dice_loss)*(total - w)/total 
    return loss

def multi_class_Dice_loss(y_true, y_pred):
    classes=3
    smooth=1.e-5
    #def dice_score(y_true, y_pred):              
    gdc=0
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    for i in range(classes):
        neuminator= tf.reduce_sum(y_true[:,:,i]*y_pred[:,:,i])
        denominator=tf.reduce_sum(y_true[:,:,i]+y_pred[:,:,i])
        gdc+=2.*(neuminator+smooth)/(denominator+smooth)
    loss=1-gdc/classes
        
    return loss#dice_score

def generalizedDice_loss(cls_weight=[1.0,1.0],classes=2):

    smooth=1.e-5

    
    def dice_score(y_true, y_pred):       
        gdc=0
        w_neu=0
        w_deno=0
        for i in range(classes):
            neuminator= tf.reduce_sum(y_true[:,:,i]*y_pred[:,:,i])
            denominator=tf.reduce_sum(y_true[:,:,i]+y_pred[:,:,i])
            if cls_weight is not None:
                w=cls_weight[i]
            else:
                w= (1/tf.square(tf.reduce_sum(y_true[:,:,i])))
        
            w = tf.where(tf.is_inf(w), tf.constant(smooth), w)
            
            w_neu+=w*neuminator
            w_deno+=w*denominator

         
        gdc=2.*(w_neu)/(w_deno)
            
        loss=1-gdc/classes
        return loss
        
    return dice_score

def tversky_loss(y_true,y_pred):
    beta=0.7
    #def loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        #return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)


def focaltversky_loss(y_true,y_pred):
    
    alpha = 0.7
    gamma = 0.75
    #def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_neg = tf.reduce_sum(y_true * (1.-y_pred))
    false_pos = tf.reduce_sum((1.-y_true)*y_pred)
        
    pt_1= (true_pos + 1)/(true_pos + alpha*false_neg + (1.-alpha)*false_pos + 1)  
    #return K.pow((1.-pt_1), gamma)
    
    return tf.math.pow((1.-pt_1), gamma)#focal_tversky

def categorical_plus_Dice(alpha=0.5,classes=5):
    
    def categorical_CCE(y_true,y_pred):
        CCE_loss=tf.keras.losses.categorical_crossentropy(y_true,y_pred)
#         CCE_loss=CCE_loss(y_true,y_pred)
        
        print('categorical cross entropy loss ', CCE_loss)   
        return CCE_loss
    
    def cce_Plus_dice(y_true,y_pred): 
        gdc=0
        smooth=1.e-5
        
        for i in range(classes):
            neuminator= tf.reduce_sum(y_true[:,:,i]*y_pred[:,:,i])
            denominator=tf.reduce_sum(y_true[:,:,i]+y_pred[:,:,i])
            gdc+=2.*(neuminator+smooth)/(denominator+smooth)
            
        dice_loss=1-gdc/classes
 
        print('Dice loss ', dice_loss)
        loss= alpha*categorical_CCE(y_true,y_pred)+(1-alpha)*dice_loss
        return loss
    
    return cce_Plus_dice 
   

            


#


    
