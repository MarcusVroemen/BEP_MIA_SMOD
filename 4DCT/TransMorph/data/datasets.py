#%%
import os, glob
import torch, sys
from torch.utils.data import Dataset
# from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from torchvision import transforms
import collections
from data import trans
# import trans 

class Lung4DCTTrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        # path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/train/image/"
        self.path = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path=self.path[0]
        paths=[]
        for subfolder in os.listdir(path):
            for filename in os.listdir(path+str(subfolder)):
                paths.append(path+str(subfolder)+"/"+str(filename))
        #paths[index*3]=T00, paths[index*3+1]=T50, paths[index*3+2]=T90
        x = nib.load(paths[index*3+1])  #T50 -> moving image
        x = x.get_fdata()
        y = nib.load(paths[index*3])  #T00
        y = y.get_fdata()
        
        # print(x.shape, y.shape)#((160, 128, 160), (160, 128, 160))
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#((1, 160, 128, 160), (1, 160, 128, 160))
        x,y = self.transforms([x, y])
        # print(x.shape, y.shape)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, 64, :], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, 64, :], cmap='gray')
        # plt.show()
        
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):#!
        path=self.path[0]
        paths=[]
        for subfolder in os.listdir(path):
            for filename in os.listdir(path+str(subfolder)):
                paths.append(path+str(subfolder)+"/"+str(filename))
                
        return int(len(paths)/3)

class Lung4DCTValDataset(Dataset): #!x_lm and y_lm added
    def __init__(self, data_path, landmark_path, transforms):
        # data_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/val/image/"
        self.path = data_path
        # landmark_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/val/landmarks/"
        self.lm_path = landmark_path
        self.transforms = transforms

    def __getitem__(self, index):
        # Landmarks
        lm_path=self.lm_path[0]
        lm_paths=[]
        for subfolder in os.listdir(lm_path):
            for filename in os.listdir(lm_path+str(subfolder)):
                lm_paths.append(lm_path+str(subfolder)+"/"+str(filename))
        x_lm = np.loadtxt(lm_paths[index*2+1])  #T50
        y_lm = np.loadtxt(lm_paths[index*2])    #T00
        # print(x_lm.shape, y_lm.shape)
        
        # Images
        path=self.path[0]
        paths=[]
        for subfolder in os.listdir(path):
            for filename in os.listdir(path+str(subfolder)):
                paths.append(path+str(subfolder)+"/"+str(filename))
                # names_individual.append(str(subfolder)+"/"+str(filename))
        x = nib.load(paths[index*3+1])  #T50 -> moving image
        x = x.get_fdata()
        y = nib.load(paths[index*3])  #T00
        y = y.get_fdata()
        
        x, y = x[None, ...], y[None, ...]#((160, 128, 160), (160, 128, 160)) -> ((1, 160, 128, 160), (1, 160, 128, 160))
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, 64, :], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, 64, :], cmap='gray')
        # plt.show()
        
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y, x_lm, y_lm

    def __len__(self):
        path=self.path[0]
        paths=[]
        for subfolder in os.listdir(path):
            for filename in os.listdir(path+str(subfolder)):
                paths.append(path+str(subfolder)+"/"+str(filename))
                
        return int(len(paths)/3)

# if __name__ == '__main__':
    # train_dir = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/train/image/"
    # train_composed = transforms.Compose([trans.RandomFlip(0),
    #                                         trans.NumpyType((np.float32, np.float32)),
    #                                         transforms.Normalize((0.5,), (0.5,))])
    # # transform = transforms.Compose([
    # #     transforms.Resize((256, 256)),
    # #     transforms.Normalize((0.5,), (0.5,)),
    # #     trans.NumpyType((np.float32, np.float32))
    # # ])
    # train_composed = transforms.Compose([trans.RandomFlip(0),
    #                                     trans.CenterCrop(50),
    #                                     trans.NumpyType((np.float32, np.float32)),])
    # train_set = Lung4DCTDataset(glob.glob(train_dir), transforms=train_composed)
    # train_set[1]

    # index=1
    
    # path=train_dir
    # paths=[]
    # for subfolder in os.listdir(path):
    #     for filename in os.listdir(path+str(subfolder)):
    #         paths.append(path+str(subfolder)+"/"+str(filename))
    #         # names_individual.append(str(subfolder)+"/"+str(filename))
    # #paths[index*3]=T00, paths[index*3+1]=T50, paths[index*3+2]=T90
    # x = nib.load(paths[index*3+1])  #T50 -> moving image
    # x = x.get_fdata()
    
    # y = nib.load(paths[index*3])  #T00
    # y = y.get_fdata()
    
    # # print(x.shape, y.shape)#((160, 128, 160), (160, 128, 160))
    # # transforms work with nhwtc
    # x, y = x[None, ...], y[None, ...]
    # print(x.shape, y.shape)#((1, 160, 128, 160), (1, 160, 128, 160))
    # x,y = train_composed([x, y])
    # #sys.exit(0)
    # x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
    # y = np.ascontiguousarray(y)
    
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(x[0, :, 64, :], cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(y[0, :, 64, :], cmap='gray')
    # plt.show()
    
    #sys.exit(0)
    #y = np.squeeze(y, axis=0)
    # x, y = torch.from_numpy(x), torch.from_numpy(y)

    # val_dir = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/val/image/"
    # val_lm_dir = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/val/landmarks/"
    # val_composed = transforms.Compose([trans.RandomFlip(0),
    #                                 # trans.CenterCrop(50),
    #                                 trans.NumpyType((np.float32, np.float32)),])
    # val_set = Lung4DCTValDataset(glob.glob(val_dir), glob.glob(val_lm_dir), transforms=val_composed)
    # data = val_set[1]
    # x_lm = data[2]
    # y_lm = data[3]
    
