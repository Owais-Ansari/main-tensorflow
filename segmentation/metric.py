'''
Created on 25-Jan-2021

@author: owaish
@co-author: Harshal
'''
'''
Background must have class label '0', Non-zero class labels are considered as foreground objects
All the class labels must be 0, 1, 2,.. and so on

'''
 
import numpy as np
import cv2
import os
import sklearn.metrics as metrics
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from Model import histNet_v2,VGG_FCN8,UNET_inception_ResNet_TooSmallSize_Modified
from metrics_Evaluation import mean_iou
from DeFUnet import DeFUnet_lite,Unet
from Post_Processing.CRF import crf
#import Functions as Fun
#import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
K.set_image_data_format('channels_first')
K.set_learning_phase(0)

# Macro Defination
# LABEL_PATH = '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/MPUH/Dr Taher/dataset 2/tiles 10x all/keras training dataset/val/val_labels_all/'
# color_Path = '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/MPUH/Dr Taher/dataset 2/tiles 10x all/keras training dataset/val/val_data_all/'
# LABEL_PATH = '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/MPUH/Dr Taher/dataset/Training set/keras training/val/val_labels_all/'
# color_Path = '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/MPUH/Dr Taher/dataset/Training set/keras training/val/val_data_all/'
LABEL_PATH = '/home/owaish/eclipse-workspace/thymus/images/val/val_labels_all/'
color_Path = '/home/owaish/eclipse-workspace/thymus/images/val/val_data_all/'
#DecoderPath = '/home/owaish/eclipse-workspace/weights/50 grades 10x/grades/UnetI3/UnetI3-320-0.907958-0.523233-0.92.h5'
#DecoderPath = '/home/owaish/eclipse-workspace/weights/50 grades 10x/grades/UnetI3/UnetI3-370-0.914622-0.522541-0.92.h5'
#DecoderPath= '/mnt/X/INTERNALPROJECTS/CHALLENGES/Prostatecancer/MPUH/Dr Taher/dataset 2/tiles 10x all/keras training dataset/histNet_v2-1131-0.239073-0.59-0.87.h5'
#DecoderPath = '/home/owaish/eclipse-workspace/weights/50 grades 10x/grades/UnetI3_512_again/histNetV2_512-90-0.856674-0.557498-0.85.h5'
DecoderPath = '/home/owaish/eclipse-workspace/thymus/images/logAndWeights/2/Unet_incepResSmallModifed-112-0.962059-0.038478-0.99.h5'
outdir = '/home/owaish/eclipse-workspace/thymus/images/logAndWeights/2/Confusion images/'
NO_OF_CLASSES = 2

height = 1024
width = 1024

extention_clr = '.png'
OUTPUT_SUFFIX = '.png'
 
confusion_matrix = np.zeros([NO_OF_CLASSES, NO_OF_CLASSES]) 
output_file_name_list = Path(LABEL_PATH).glob('*' + OUTPUT_SUFFIX)

def sampleMeanStdExcludeWhite(img):
    img  = img.astype("float32")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    x, y = np.where(imgGray < 256)
   
    b_ch_mean=np.mean(img[x,y,0])
    g_ch_mean=np.mean(img[x,y,1])
    r_ch_mean=np.mean(img[x,y,2])
   
    b_ch_std=np.std(img[x,y,0])
    g_ch_std=np.std(img[x,y,1])
    r_ch_std=np.std(img[x,y,2])
   
    img[:, :, 0] = (img[:, :, 0] - b_ch_mean)/b_ch_std
    img[:, :, 1] = (img[:, :, 1] - g_ch_mean)/g_ch_std
    img[:, :, 2] = (img[:, :, 2] - r_ch_mean)/r_ch_std
    return img


def testBatch(Tile,model):          
    im = Tile
    #Compute the mean for data normalization
#     b_ch=np.mean(im[:,:,0])
#     g_ch=np.mean(im[:,:,1])
#     r_ch=np.mean(im[:,:,2])  
#     # Mean substraction  
#         
#     im_ = np.array(im, dtype=np.float32)                             
#     im_ -= np.array((b_ch,g_ch,r_ch))
#       
#     #compute the standard deviation
#     b_ch=np.std(im[:,:,0])
#     g_ch=np.std(im[:,:,1])
#     r_ch=np.std(im[:,:,2])
#     im_ /= np.array((b_ch,g_ch,r_ch))
#     #im_ = im_.astype('uint8')
    im_ = sampleMeanStdExcludeWhite(im)
    
    data = []
    #data.append(im_)
    data.append(np.rollaxis((im_),2))
 
    temp = np.array(data)     
    prob = model.predict(temp, verbose= 1)
 
    prediction = np.argmax(prob[0],axis=1)
    prediction = np.reshape(prediction,(width,height))  
    #print(prediction.shape) 
    scale = np.uint8(255/(NO_OF_CLASSES-1)) #converting hot encoding back to labels
    norm_image = scale*np.uint8(prediction)
    
#     print(np.unique(norm_image))
    return norm_image
def label_cleaner(lab):
    lab[lab==0]=0
#     lab[lab==63]=1
#     lab[lab==126]=2
#     lab[lab==189]=3
#     lab[lab==252]=4
    lab[lab==255]=1
    return lab

def FP_FN(img1,img2,outdir,treshold = 0.07):
    '''Only valid for binary class 0 and 1 please make adjustment here if you want to use it for multiclass
    img1: Ground truth
    img2: Predicted
    '''
    xFP,yFP = np.where((img1==0) and (img2==1))
    xFN,yFN = np.where((img1==1) and (img2==0))
    img2[img2==1]= 255
    img2[xFN,yFN]= 150
    img2[xFP,yFP]= 55
    tot_FGround_pixel = cv2.countNonZero(img1[img1==255]) # total foreground pixel in ground truth image
    cutoff = int(tot_FGround_pixel*treshold)
    FP_pixel = cv2.countNonZero(img1[img1==55])
    FN_pixel = cv2.countNonZero(img1[img1==150])
    
    if (FP_pixel >=cutoff or FN_pixel>=cutoff):
        return cv2.imwrite(outdir,img2)
    
    
   
def LoadModel():
    imheight = height
    imwidth =  width
    imdepth = 3
    classes = NO_OF_CLASSES
    #model = histNet_v1(imwidth,imheight,imdepth, classes, None,'channels_last', var.DecoderPath)
    #model =  VGG_FCN8(imwidth, imheight, imdepth, classes, weightsPath=DecoderPath)
    model = UNET_inception_ResNet_TooSmallSize_Modified(imwidth, imheight, imdepth, classes, weightsPath=DecoderPath)
    #model = Unet(imwidth, imheight, imdepth, classes, classes, data_format = 'channels_last', retrain='False', weightsPath=DecoderPath)
    #model= getattr(Model,var.ModelFunctionName)(imwidth, imheight, imdepth, classes, None,'channels_last',var.DecoderPath)    
    model.summary()
    return model

model = LoadModel()

for output_file_name in output_file_name_list:
    #print(output_file_name)
    
    gtbase= os.path.basename(str(output_file_name))
    print(gtbase)
    color_img = color_Path+gtbase[:-4]+extention_clr
    output_image_clr = cv2.imread(color_img)
    #output_image = resize(output_image,(width,height), mode = 'constant', preserve_range =True)
    output_image = testBatch(output_image_clr,model)
    output_image = label_cleaner(output_image)
    
    output_image_clr = cv2.cvtColor(output_image_clr, cv2.COLOR_BGR2GRAY)
    
    #ground_truth = cv2.imread(output_file_name,0)
    ground_truth = np.asarray(Image.open(output_file_name).convert('L'), dtype=np.uint8)
    output_image = crf(output_image_clr,output_image)
    #ground_truth = resize(ground_truth,(width,height), mode = 'constant', preserve_range =True)
    #ground_truth = cv2.resize(ground_truth,(512,512), interpolation = cv2.INTER_AREA)
    confusion_matrix += metrics.confusion_matrix(ground_truth.reshape(-1),output_image.reshape(-1), range(NO_OF_CLASSES))
    #
    outDIR = outdir + gtbase
    FP_FN(ground_truth,output_image,outDIR)
    
    #print(confusion_matrix)
 
 
total_predictions = np.sum(confusion_matrix)
mean_accuracy = mean_iou = mean_dice = 0
FP=0
FN=0

for class_id in range(1, NO_OF_CLASSES):
#    tn, fp, fn, tp = confusion_matrix.ravel()
    #cm = confusion_matrix(ground_truth.reshape(-1),output_image.reshape(-1), range(NO_OF_CLASSES))
#     tp = np.diag(confusion_matrix)
#     fp = np.sum(confusion_matrix, axis=0) - tp
#     fn = np.sum(confusion_matrix, axis=1) - tp
#     tn = np.sum(confusion_matrix)-(tp+fp+fn)
#     iou = (tp) / (tp + fp + fn)
#     dice = (2 * tp) / (2 * tp + fp + fn)
    tp = confusion_matrix[class_id, class_id]
    fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1 :, class_id])
    fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1 :])
    tn = total_predictions - tp - fp - fn
    accuracy = (tp + tn) / (tn + fn + tp + fp) 
    mean_accuracy += accuracy
 
    if ((tp + fp + fn) != 0):
        iou = (tp) / (tp + fp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
    else:
        # When there are no positive samples and model is not having any false positive, we can not judge IOU or Dice score
        # In this senario we assume worst case IOU or Dice score. This also avoids 0/0 condition
        iou = 0.0
        dice = 0.0
 
    mean_iou += iou
    mean_dice += dice
    FP+=fp 
    FN+=fn

    print("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {}:".format(class_id, accuracy, iou, dice))

mean_accuracy = mean_accuracy / (NO_OF_CLASSES - 1)
mean_iou = mean_iou / (NO_OF_CLASSES - 1)
mean_dice = mean_dice / (NO_OF_CLASSES - 1)
FP= FP/(NO_OF_CLASSES - 1)
FN= FN/(NO_OF_CLASSES - 1)

print("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {},FP: {},FN: {}".format(mean_accuracy, mean_iou, mean_dice,FP,FN))


