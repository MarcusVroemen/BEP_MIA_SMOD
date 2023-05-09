import os, glob
import torch, sys
from torch.utils.data import Dataset
from data.data_utils import pkload #!removed .
import matplotlib.pyplot as plt
from data import trans
import numpy as np
from torchvision import transforms


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 100], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y[0, :, :, 100], cmap='gray')
        # plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(x[0, :, :, 150], cmap='gray')
        # plt.subplot(2, 2, 2)
        # plt.imshow(y[0, :, :, 150], cmap='gray')
        # plt.subplot(2, 2, 3)
        # plt.imshow(x_seg[0, :, :, 150], cmap='gray')
        # plt.subplot(2, 2, 4)
        # plt.imshow(y_seg[0, :, :, 150], cmap='gray')
        # plt.show()
        
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
    

# atlas_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/atlas.pkl'
# train_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/Train/'
# val_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/Val/'

# train_composed = transforms.Compose([trans.RandomFlip(0),
#                                         trans.NumpyType((np.float32, np.float32)),])
# val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
#                                     trans.NumpyType((np.float32, np.int16))])
# train_set = IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
# val_set = IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
# # train_set[1]
# val_set[1]