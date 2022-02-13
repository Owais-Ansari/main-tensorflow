import os

import numpy as np
from tensorflow import keras
#from skimage.transform import resize
import cv2
from PIL import Image

from albumentations import *



#global width,height
def get_aug(p=1.0):
    return Compose([
        #Resize(512,512),
         #HorizontalFlip(),
         #VerticalFlip(),
         RandomRotate90(),
#         # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.15*p, 
#                         # border_mode=cv2.BORDER_REFLECT),
#          OneOf([
#              OpticalDistortion(p=0.1),
#              GridDistortion(p=0.1),
# #             IAAPiecewiseAffine(p=0.3*p),
#          ], p=p), #0.3
         OneOf([
#             #RandomSizedCrop(min_max_height=(128, 51),height=1024, width=1024, p=0.1),
#             #PadIfNeeded(min_height=original_height, min_width=original_width, p=0.05),
             #RandomContrast(),
             #Blur(blur_limit=3, p=0.1)
             ],p=0.2),
         OneOf([
              HueSaturationValue(15,25,15),
              CLAHE(clip_limit=0.8),
              RandomGamma(),
              RandomBrightnessContrast()
         ], p=0.4),
    ],p=p)
    
def get_aug_val(p=1):
    return Compose([
        #Resize(512,512)
        ],p=p)
    
    
def sampleMeanStdExcludeWhite(img):
    
    img  = img.astype("float32")
   
    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = Image.fromarray(img).convert('L')
    imgGray = n.asarray(imgGray)
    x, y = np.where(imgGray <256)
   
    r_ch_mean=np.mean(img[x,y,0])
    g_ch_mean=np.mean(img[x,y,1])
    b_ch_mean=np.mean(img[x,y,2])
   
    r_ch_std=np.std(img[x,y,0])
    g_ch_std=np.std(img[x,y,1])
    b_ch_std=np.std(img[x,y,2])
   
    img[:, :, 0] = (img[:, :, 0] - r_ch_mean)/r_ch_std
    img[:, :, 1] = (img[:, :, 1] - g_ch_mean)/g_ch_std
    img[:, :, 2] = (img[:, :, 2] - b_ch_mean)/b_ch_std
    
    return img
   

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root, split, batch_size=32, dim=(32,32), n_channels=3,
                 n_classes=10, shuffle=True,tfms=None):
        'Initialization'
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.tfms = tfms
        
        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')
        
        #file_list = os.path.join(self.root, self.split + '_segmentation', self.split + ".txt")
        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list,'r'))]
        #self.files = [line.rstrip() for line in  open(file_list,'r')]
        

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, self.n_channels, *self.dim))
        X = np.empty((self.batch_size,*self.dim,self.n_channels))
        #y = np.empty((self.batch_size, *self.dim), dtype=int)
        #y = np.empty((self.batch_size, self.dim[0]*self.dim[1], self.n_classes), dtype=int)
        y = np.empty((self.batch_size, self.dim[0]*self.dim[1], self.n_classes), dtype=int)
        # Generate data
        for i, image_id in enumerate(list_IDs_temp):
#            image_path = os.path.join(self.image_dir, image_id + '.png')
            image_path = os.path.join(self.image_dir, image_id + '.png')
            label_path = os.path.join(self.label_dir, image_id + '.png')
                        
            image =  np.asarray(Image.open(image_path), dtype=np.uint8)
            label = np.asarray(Image.open(label_path).convert('L'), dtype=np.uint8)
            #labels = np.zeros([label.shape[0],label.shape[1]],dtype='uint8')
            #label = labels.copy()
            
            #print(np.unique(label))
            
            if self.tfms is not None:
                augmented = self.tfms(image=image, mask=label)
                image, label = augmented['image'], augmented['mask']

            #normalization 
            im = image
            #im_ =  sampleMeanStdExcludeWhite(im)
            #im_ = (255-image)/255
         # Global_Mean computation of each channel and subtract from it
         # imagenet mean and image net std deviation
            b_ch=103.939#211.96
            g_ch=116.779#184.93
            r_ch=123.680#206.67
            
            b_ch_std = 21.588
            g_ch_std = 49.67
            r_ch_std = 26.84
            
            im_ = np.array(im, dtype=np.float32)
            im_ -= np.array((r_ch,g_ch,b_ch))
            im_ /= np.array((r_ch_std,g_ch_std,b_ch_std))
                
            im_  = im_.astype("float32") 
        #Individual channel-wise mean substraction
#             im_ -= np.array((b_ch,g_ch,r_ch))           
            #im_ = np.rollaxis((im_),2) # swap the axis, output=(3,512,512)
            X[i,] = im_

            #Store class
            label = binarylabel(label, self.n_classes)
            #print('label- After binaryFunction',label.shape) #(512,512,3)
            label = np.reshape(label, (self.dim[0]*self.dim[1], self.n_classes))

            y[i,] = label

        return X,y
    
     


def binarylabel(im_label,classes):
    
    im_dims = im_label.shape
    
    lab=np.zeros([im_dims[0],im_dims[1],classes],dtype="uint8")
    for class_index in range(classes):
        lab[im_label==class_index, class_index] = 1
    #print(lab.shape)  # = (512,512,3)  
    return lab
    
    