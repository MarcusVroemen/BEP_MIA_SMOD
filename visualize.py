import os
import numpy as np
# from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

base_path = "C:\\Users\\20203531\\OneDrive - TU Eindhoven\\Y3\\Q4\\BEP\\code\\4DCT\\"
path_image = base_path + "train\\image\\"
path_landmarks = base_path + "train\\landmarkt\\"

#%%
path_gz_file = path_image+'case_001\\T00.nii.gz'
img = nib.load(path_gz_file)

# print(img)

img= img.get_fdata()
plt.imshow(ndi.rotate(img[:,80, :],-90), cmap='gray')


#%%
plt.imshow(ndi.rotate(img[:,127, :],-90), cmap='gray')
plt.imshow(ndi.rotate(img[139,:, :],-90), cmap='gray')
plt.imshow(ndi.rotate(img[:,:, 142],-90), cmap='gray')