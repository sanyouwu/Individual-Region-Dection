import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # dataset module
import torchvision.models as models
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import random
import PIL
from PIL import Image
import time
import numpy as np

def convert_data(X_train,Y_train,X_test,Y_test,batch_size,augmentation =  False, tv = None):
    ## tv: total variation
    transform_train = transforms.Compose([
        transforms.ToPILImage(),

        # transforms.Resize((img_shape_m,img_shape_n)),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomFlip(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])


    # @staticmethod
    def augmentation(data):
        '''
        do augmentation within a batch
        '''
        tmp = map(transform_train, torch.tensor(data).unbind(dim = 0))
        # tmp = list(map(np.asarray, tmp))
        res = [np.array(x).transpose(2,0,1) for x in tmp]
        return np.asarray(res)

    class Labeled_dataset(torch.utils.data.Dataset):
        def __int__(self, X, Y, target_type = "test"):
            self.X = X
            self.Y = Y
            self.type = target_type

        def __getitem__(self, index):
            x = self.X[index]
            ## x shape: 100 x 3 x 50 x 50
            ## x type: np.array            
            if self.type  == "train" and self.augmentation == True:
                x = augmentation(x)
            y = self.Y[index]

            return x, y

        def __len__(self):
            return len(self.Y)

    data_train = Labeled_dataset()
    data_train.X = X_train
    data_train.Y = Y_train
    data_train.type = "train"
    data_train.augmentation = augmentation

    # labeled validation dataset
    data_test = Labeled_dataset()
    data_test.X = X_test
    data_test.Y = Y_test
    data_test.type = "test"
    # data_train.augmentation = False
    #     batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size= batch_size,shuffle = True) #shuffle=True
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size= batch_size,shuffle = False) # ,
    return train_loader,test_loader