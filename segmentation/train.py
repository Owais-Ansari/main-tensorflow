import sys
# sys.path.insert(0, '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/Data/Kaggle_Inference_Scripts/segmentation')

import numpy as np
import os
#import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input
from tensorflow.keras import callbacks 


#from HistNet import histNet_v2 #UNET_inception_ResNet_TooSmallSize_Modified
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from loss_function_for_segmentation import weighted_categorical_crossentropy_dice, multi_class_Dice_loss,focaltversky_loss#,categorical_focal_loss
import tensorflow.keras.backend as K
from tensorflow import keras
############################################### importing datagenerator############
from keras_dataloader import DataGenerator,get_aug, get_aug_val
####################### importing model ##########################################
#from DeepLab.DeepLabEff_TF import Deeplabv3_eassp
###########################################################################
#Mixed Precision
###########################################################################
from tensorflow.python.keras import mixed_precision
#tf.keras.mixed_precision.LossScaleOptimizer
from segmentation_model import Models,Deeplabv3_eassp,Deeplabv3Plus_resnet#,UEfficientNet_customize#, Deeplabv3#,Models
#Enable XLA
#tf.config.optimizer.set_jit(True)
# Enable AMP
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16',loss_scale="dynamic")
# mixed_precision.set_global_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)
############################################################################


from one_cycle_policy.clr import OneCycleLR,LRFinder


################### GPU configuration ###################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE"] = "0"
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# np.random.seed(1337)  # for reproducibility
# 
# 
# 
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# set_session(tf.Session(config=config))

# tensorboard make dire
#log_dir = '/mnt/X/Owaish/from_sai/Prostate_gleason/MPUH/TensorBoard/histNet_v1_model/'
#log_dir = '/mnt/X/Owaish/from_sai/Prostate_gleason/MPUH/TensorBoard/Unet_SqNEt_model/log/unet/'
log_dir = '/home/owaishs/data/TransU/ralp_cri/'
#log_dir = '/home/owaishs/data/Weights/Dinisha_ralph/Binary models/cribriform/'

# if not exists(dirs):
#    os.makedirs(dirs)
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)


#dice_function
def dice_coef(y_true, y_pred):
    return (2. * tf.reduce_sum(y_true * y_pred) + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 50, 150, 350, 500, 1000 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 0.0005
#     if epoch >= 50 and epoch < 150:
    if epoch >=25  and epoch < 50:
        lr *= 1e-1
    elif epoch >= 50 and epoch <70:
        lr *= 1e-1
    elif epoch >= 40 and epoch <500:
        lr *= 1e-1
    elif epoch >= 500 and epoch <1000:
        lr *= 1e-4
    elif epoch >= 1000:
        lr *= 1e-5
    print('Learning rate: ', lr)
    return lr



#root_folder = '/mnt/X/Owaish/Prostate gleason/Pattern detection labels/Tiles for train and val/'
#root_folder='/mnt/store01/MPUH/Data_with_benign_added_from_new93images/GradesplusGrades/train_val_data/'
#root_folder = '/home/owaishs/data/'
root_folder = '/mnt/imgproc/Dinisha/GleasonGrading/G4G5Pattern/5X/Training/'
#root_folder = '/mnt/X/Pranab_Samanta/ProstateRALP/clean_dataSet_Radboud/'
#classes = 5 # current dataset num_classes
classes = 3



#path2write = '/mnt/X/Owaish/from_Dev/Organ Separation/Model_weights/'
#path2write = '/mnt/X/Owaish/from_sai/Prostate_gleason/MPUH/Model_weight_unet_sq/'
path2write = log_dir
#filepath=path2write+ 'UNET_SqNET-{epoch:02d}-{val_loss:02f}-{val_dice_coef:.2f}-{val_mean_iou:02f}.h5'
filepath=path2write+'Unet3plus-{epoch:02d}-{loss:02f}-{val_loss:02f}-{val_accuracy:02f}-{val_dice_coef:02f}.h5'
#path2write = '/mnt/X/Dev Kumar/PanIN/20x_mag_data/Third_iteration/Data/data_UP/histnet_model/model/'
#filepath=path2write+'Unet_best.h5'
#preTrained =  '/home/owaishs/data/Weights/Dinisha_ralph/BinaryCRI/Unet++-25-0.220353-0.234080-0.969966-0.969931.h5'
# new_classes=6 # pretrained model num_classespr
preTrained = None# '/home/owaishs/data/TransU/ralp_cri/TransUnet-12-0.094358-0.114642-0.960894-0.955556.h5'
#preTrained = '/mnt/X/Owaish/temp/Model and script/histNet_v2-954-0.125376-0.76-0.95.h5'
new_classes = None#new classes if model is trained from the previous model weights having similar data with some new classes

#Define the Image Size
batchsize = 4
depth= 3
height = 1024
width = 1024

# lr_manager = OneCycleLR(num_samples=1650,batch_size=batchsize,max_lr=0.0000189,
#                  end_percentage=0.1,
#                  scale_percentage=None,
#                  maximum_momentum=None,
#                  minimum_momentum=None,
#                  verbose=True)

# lr_finder = LRFinder(num_samples=1650, batch_size=batchsize, minimum_lr=1e-8, maximum_lr=0.001,
#                      lr_scale='linear',
#                      # validation_data=(X_test, Y_test),  # use the validation data for losses
#                      validation_sample_rate=5,
#                      save_dir=path2write, verbose=True)
#LRFinder.plot_schedule()
starter_learning_rate = 0.001
end_learning_rate = 0.00005
decay_steps = 10000

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.33)


#model  = Models(width,height,depth,classes, model_type = 'unet_3plus_2d')

#model = Unet(width, height, depth, classes, new_classes, data_format='channels_last', retrain='False', weightsPath=preTrained)
#model = UNET_inception_ResNet_TooSmallSize_Modified(width, height, depth, classes, weightsPath=preTrained)
#model = Deeplabv3(input_shape=(width, height, depth), classes=classes, OS=16, backbone= 'xception',activation='softmax')
#model = Deeplabv3Plus_resnet(input_shape=(width, height, depth), classes=classes,OS=16, activation = 'softmax')
model = Deeplabv3_eassp( input_shape=(width, height, depth), classes=classes, backbone = 'efficientnetb0')
#model = UEfficientNet_customize(input_shape=(width, height, depth),classes = classes, backbone = 'efficientnetb0', e_assp = False)
#model =  Models(width,height,depth,classes, model_type = 'unet_3plus_2d',backbone = 'EfficientNetB0')

#model = Models(width,height,depth,classes, model_type= 'transunet_2d')
#model.load_weights(preTrained)

# for indx, layer in enumerate(model.layers):
#     if indx > 158:
#         layer.trainable = False
#     else:
#         layer.trainable = True
# #model = DeFUnet_lite(width, height, depth, classes, weightsPath=preTrained)
#model= histNet_v2(width, height, depth, classes, new_classes, data_format = 'channels_last', retrain='False',weightsPath=None)
print(len(model.layers))
print(model.summary())

##=================================Freezing batchnorm ====================================================================================

# for indx, layer in enumerate(model.layers):
#     if layer.name =='block6a_expand_activation':
#         layer_index = indx

    # if isinstance(layer, BatchNormalization):
    #     layer.trainable = False
    # else:
    #     layer.trainable = True


  # for layer in model.layers:
  #   if layer.name.startswith('bn'):
  #     layer.call(layer.input, training=False)
#============================================================================================================================================




modelCheck = callbacks.ModelCheckpoint(filepath, monitor = 'val_loss', verbose=1, save_best_only = False, save_weights_only = False, mode='auto', period=1)


# step = tf.Variable(0, trainable=False)
# schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
#     [10000, 15000], [1e-0, 1e-1, 1e-2])
# # lr and wd can be a function or a tensor
# lr = 1e-1 * schedule(step)
# wd = lambda: 1e-4 * schedule(step)
#
# # ...
#
# opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
opt = Adam()
#opt = SGD(momentum=0.93,nesterov=True, name='SGD')
#opt = tf.compat.v1.train.AdamOptimizer()
# Uncomment to for wrapping optimizer for mixed precision
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt,loss_scale="dynamic")
#opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt)

#from tensorflow.python.keras import mixed_precision
#Parameters
params = {'dim': (height, width),
          'batch_size': batchsize,
          'n_classes': classes,
          'n_channels': depth,
          'shuffle': True,
          'tfms':get_aug(p=1)}

params_val = {'dim': (height, width),
          'batch_size': batchsize,
          'n_classes': classes,
          'n_channels': depth,
          'shuffle': True,
          'tfms':get_aug_val(p=1)}

# Generators
training_generator = DataGenerator(root_folder, 'train',  **params)

validation_generator = DataGenerator(root_folder, 'val',  **params_val)

print("Compiling Model...")

# Set the compiler parameter for the training
#model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[dice_coef,"accuracy"], sample_weight_mode='auto')
#losss = tf.keras.losses.CategoricalCrossentropy() # multi_class_Dice_loss
# focaltversky_loss
# multi_class_Dice_loss
# weighted_categorical_crossentropy_dice multi_class_Dice_loss
model.compile(loss = multi_class_Dice_loss, optimizer = opt, metrics = [dice_coef,'accuracy'], sample_weight_mode='auto')
lr_scheduler = LearningRateScheduler(lr_schedule)

#lr_scheduler = LearningRateScheduler(learning_rate_fn)

#Train the Network
#===============================================================================
#Splitting dataset into training and testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.25, random_state=10)
#training_generator1,validation_generator = train_test_split(training_generator,test_size=0.25, random_state=10)


print("Training the Model...")
#results = model.fit_generator(generator = (X_train,y_train), validation_data = (X_test,y_test), epochs=201, verbose=2, use_multiprocessing=True,
#                    workers=6, callbacks=[modelCheck,lr_scheduler, tbCallBack])
results = model.fit_generator(generator = training_generator, validation_data = validation_generator, epochs = 50, 
                              verbose=11, use_multiprocessing = False, workers=12, callbacks=[modelCheck,lr_scheduler,tbCallBack])


                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                               