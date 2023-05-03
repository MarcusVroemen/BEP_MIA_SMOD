#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
# from skimage import io 

# Define the path to the images
base_path = "C:\\Users\\20203531\\OneDrive - TU Eindhoven\\Y3\\Q4\\BEP\\code\\4DCT\\"
path_image = base_path + "train\\image\\"
path_landmarks = base_path + "train\\landmarkt\\"
path_gz_file = path_image+'case_001\\T00.nii.gz'

#%%
gzip_file_data_frame = pd.read_csv(
    path_gzip_file, compression='gzip',
    header=0, sep=',', quotechar='"')

print(gzip_file_data_frame.head(5))


#%%
#==================================
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
#==================================
# load image (4D) [X,Y,Z_slice,time]
nii_img = nib.load(path_gz_file)
nii_data = nii_img.get_fdata()

fig, ax = plt.subplots(number_of_frames, number_of_slices,constrained_layout=True)
fig.canvas.set_window_title('4D Nifti Image')
fig.suptitle('4D_Nifti 10 slices 30 time Frames', fontsize=16)
# %%
