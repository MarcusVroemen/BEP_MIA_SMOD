import os
import numpy as np
# from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

#%% Append all paths for training data
path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/train/image/"
path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/val/image/"
path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/test/image/"
paths=[]
names=[]
for subfolder in os.listdir(path):
    for filename in os.listdir(path+str(subfolder)):
        paths.append(path+str(subfolder)+"/"+str(filename))
        names.append(str(subfolder)+"/"+str(filename))

#%% Plot 3 slice directions and all phases        
select_case = 0 
paths=paths[select_case*3:select_case*3+3]
names=names[select_case*3:select_case*3+3]
fig, axs = plt.subplots(len(paths),3, figsize=(30*len(paths),90))
for i in range(len(paths)):
    img = nib.load(paths[i])
    img = img.get_fdata()
    for j in range(3):
        if j==0: #saggital #(160, 128, 160)
            axs[i,j].imshow(ndi.rotate(img[60,:, :],-90), cmap='gray')
        elif j==1: #coronal
            axs[i,j].imshow(ndi.rotate(img[:,64, :],-90), cmap='gray')
            axs[i,j].set_title(str(names[i]), fontsize=100)  
        elif j==2: #axial
            axs[i,j].imshow(ndi.rotate(img[:,:, 80],-90), cmap='gray')
        plt.tight_layout()
        axs[i,j].axis("off")

#%% Plot
select_image = 0  
fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
img = nib.load(paths[select_image])
img = img.get_fdata()
n_slice = img.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, im in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(img[im, :, :], -90), cmap='gray')
    axs.flat[idx].axis('off')     
plt.tight_layout()
plt.show()



#%%
from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
# %% IXI DATASET
atlas_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/atlas.pkl'
train_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/Train/'
val_dir = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/IXI_data/Val/'

train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

x, y = train_set[1]
# plt.imshow(x[0][100,:, :], cmap='gray')
# plt.imshow(y[0][100,:, :], cmap='gray')
# %%
# path = self.paths[index]
#         x, y = pkload(path)