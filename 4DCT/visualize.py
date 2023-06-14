#%%
import os
import numpy as np
# from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

#%% Append all paths for training data
data_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/{}/image/"
data_type="test"
select_case = 0 
path = data_path.format(data_type)
paths=[]
names=[]
for subfolder in os.listdir(path):
    for filename in os.listdir(path+str(subfolder)):
        paths.append(path+str(subfolder)+"/"+str(filename))
        names.append(str(subfolder)+"/"+str(filename))

#%% Plot 3 slice directions and all phases        
paths_case=paths[select_case*3:select_case*3+3]
names=names[select_case*3:select_case*3+3]
fig, axs = plt.subplots(len(paths_case),3, figsize=(30*len(paths_case),90))
for i in range(len(paths_case)):
    img = nib.load(paths_case[i])
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
data_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/deformationField_case026.nii.gz"
data_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/deformationField.nii.gz"
img = nib.load(data_path)
img = img.get_fdata()
x=img[:,:,:,0,0]
slice_img = img[60,:,:,0,0]
plt.imshow(x[60,:,:], cmap='gray')
# %%
