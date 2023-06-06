# %%
import itk
# Import the required libraries
# import elastix
# import imageio
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
import seaborn as sns
# import scipy.stats as stats
# import openpyxl
import nibabel as nib
import random
import os
import sys
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import cv2
import scipy.ndimage as ndi
import gryds
from sklearn import preprocessing
from sklearn.decomposition import PCA

import datasets_utils as DU 
import elastix_functions as EF
plt.rcParams['image.cmap'] = 'gray'


#%% Data preparation
root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
train_dataset = DU.DatasetLung('train', root_data=root_data, augment_def=False, phases="in_ex")

img_data_T00, img_data_T50, img_data_T90 = [], [], []
for i in range(len(train_dataset)):
    img_fixed, img_moving,_,_ = train_dataset[i]
    if i%2==0:
        img_data_T00.append(img_fixed.squeeze(0).numpy()), img_data_T50.append(img_moving.squeeze(0).numpy())  #fixed T50 are dubble, moving T00 and T90       
    else:           
        img_data_T90.append(img_fixed.squeeze(0).numpy())

img_data_T00 = [itk.image_from_array(arr) for arr in img_data_T00]
img_data_T50 = [itk.image_from_array(arr) for arr in img_data_T50]
img_data_T90 = [itk.image_from_array(arr) for arr in img_data_T90]

# Create an image grid
img_grid = np.zeros((160, 128, 160))
line_interval = 5  # Adjust this value to change the interval between lines
img_grid[:, ::line_interval, :] = 1
img_grid[:, :, ::line_interval] = 1
img_grid[::line_interval, :, :] = 1
# plt.imshow(img_grid[:,64,:])

#%% STEP 1 ATLAS GENERATION (or load)
generate=False
if generate:
    # Generate atlas image
    img_atlas = EF.generate_atlas(img_data=img_data_T00)
    plt.imshow(img_atlas[:,64,:], cmap="gray")

    img = nib.load(root_data+'train/image/case_001/T00.nii.gz')
    img_atlas_nib = nib.Nifti1Image(img_atlas, img.affine)
    nib.save(img_atlas_nib, os.path.join(root_data,'atlas', 'atlasv1.nii.gz'))
else:
    img_atlas = nib.load(root_data+'atlas/atlasv1.nii.gz')
    img_atlas = img_atlas.get_fdata()
    img_atlas = itk.image_from_array(ndi.rotate((img_atlas).astype(np.float32),0)) # itk.itkImagePython.itkImageF3
    plt.imshow(img_atlas[:,64,:], cmap="gray")

#%% VALIDATION: Apply all DVFT00 to all training images
def validateDVFT00s(img_data, img_atlas, method="affine"):
    """Registrate img_data to atlas image and apply those DVFs to all img_data
    The goal of this function is to see how see how a DVF from one image to atlas 
    manipulates a different image"""
    img_data = img_data_T00[0:4]
    DVFs_params = []
    DVFs = []
    for i in range(len(img_data)):
        moving_image = img_data[i]
        result_image, DVF, result_transform_parameters = EF.registration(
                    fixed_image=img_atlas, moving_image=moving_image, 
                    method=method, plot=True)  # ?Affine or  bspline
        DVFs.append(DVF)
        DVFs_params.append(result_transform_parameters)

    # apply every transformation from original to atlas, to each image
    # this will test how DVFs from other image pairs change existing images
    for moving_image in img_data:
        for result_transform_parameters in DVFs_params:
            result_image_transformix = itk.transformix_filter(
                moving_image,
                result_transform_parameters)

            fig, axs = plt.subplots(2)
            axs[0].imshow(moving_image[:, 64, :], cmap='gray')
            axs[0].set_title("original image")
            axs[1].imshow(result_image_transformix[:, 64, :], cmap='gray')
            axs[1].set_title("image with DVFT00")
            plt.tight_layout()
            plt.show()
            
# validateDVFT00s(img_data=img_data_T00, img_atlas=A, method="affine")
# conclusion: affine gives ok results


#%% STEP 2 DVF GENERATION #################################################################
def register_to_atlas(img_data, img_atlas):
    """Generate DVFs from set images registered on atlas image"""
    DVFs_list = []
    
    for i in range(len(img_data)):
        result_image, DVF, result_transform_parameters = EF.registration(
                    fixed_image=img_atlas, moving_image=img_data[i], 
                    method="affine", plot=True)  # ?Affine or  bspline
        
        DVFs_list.append(DVF)
    
    return DVFs_list

def DVF_conversion(DVF_itk, plot=False): 
    """Converts DVF from itk to DVF usable with gryds"""
    # Reshapre DVF from (160, 128, 160, 3) to (3, 160, 128, 160)
    reshaped_DVF = np.transpose(np.asarray(DVF_itk), (3, 0, 1, 2))  
    
    # DVF elastix in pixels while DVF gryds in proportions. Therefore scale each direction with their pixelsize. 
    # DVF[0] works medial-lateral on 160 pixels, DVF[2] cranial-caudial on 160 pixels, DVF[1] posterior-anterior in 128 pixels
    DVF_scaled = np.asarray([reshaped_DVF[0]/160, reshaped_DVF[1]/128, reshaped_DVF[2]/160])
    
    # gryds transformation takes DVFs in different order: CC, AP, ML
    DVF_gryds = [DVF_scaled[2], DVF_scaled[1], DVF_scaled[0]]

    return DVF_gryds

def transform(DVF_itk, img_moving):
    DVF_gryds = DVF_conversion(DVF_itk, plot=True)
    a_bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
    an_image_interpolator = gryds.Interpolator(img_moving)
    a_deformed_image = an_image_interpolator.transform(a_bspline_transformation)
    EF.plot_registration(fixed_image=img_moving, moving_image=a_deformed_image, deformation_field=DVF_itk, full=True,
                            result_image=np.asarray(img_moving) - np.asarray(a_deformed_image), title="transformation with averaged atlas DVF",
                            name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")

#%% Generate atlas DVFs
DVFs = register_to_atlas(img_data=img_data_T00, img_atlas=img_atlas)
DVFs_arrays = [np.asarray(DVF) for DVF in DVFs]

#%% apply DVFmean to images
# DVF_mean = sum(DVFs_arrays) / len(DVFs_arrays)
# DVF_gryds = DVF_conversion(DVF_mean, plot=True)

# for img_moving in img_data_T00:
#     a_bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
#     an_image_interpolator = gryds.Interpolator(img_moving)
#     a_deformed_image = an_image_interpolator.transform(a_bspline_transformation)
#     EF.plot_registration(fixed_image=img_moving, moving_image=a_deformed_image, deformation_field=DVF_mean, full=True,
#                          result_image=np.asarray(img_moving) - np.asarray(a_deformed_image), title="transformation with averaged atlas DVF",
#                          name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")

#%% #!--------PCA-----------
#Turn each DVF into an array and reshapeto 3276800x3
DVFs_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]    
#Concatenate DVFs into matrix V with shape 3276800x(3*len(DVFs_arrays))
V = np.concatenate(DVFs_columns, axis=1)                
#Average DVFs into matrix Vmean with shape 3276800x3        
Vmean = sum(DVFs_columns) / len(DVFs_columns)             
V_mean = np.mean(V, axis=1) #?
# U eigenvectors of principal modes
# d eigenvalues of principal modes
# x array of random numbers following distribution N(0,simga)
# vg = Vmean + U*x*d
#? scale or not?

#%% #*Scale data
standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
V_standardscaled = standard_scalar.fit_transform(V)
fig, axes = plt.subplots(1, 2, figsize=(10,5))
sns.boxplot(data=V, ax=axes[0]).set(title='V', xlabel='Variable')
sns.boxplot(data=V_standardscaled, ax=axes[1]).set(title='V Standard scaled', xlabel='Variable')
plt.tight_layout()
plt.show()
#%% #*Determine number of components needed
def determine_n_PC(data):
    """ Function that plots the variance explained by the principal components
    per number of components used. This can be used to the determine the components
    necessary to explain for example 90% of the data's variance.
    """
    variances = []
    PCs = []
    # For each number of components calcualate explained variance
    for i in range(1,len(data[1])+1):
        pca = PCA(n_components=i) # estimate only i PCs
        pca.fit_transform(data) # project the original data into the PCA space
        variance_explained = (pca.explained_variance_ratio_).sum()
        variances.append(variance_explained)
        PCs.append(i)
    
    # Plot this data in a barplot    
    fig = plt.figure()
    plt.title('Variance explained by PCA')
    plt.bar(PCs, variances)
    plt.xlabel('Number of principle components')
    plt.ylabel('Variance explained')
    plt.xticks(PCs)
    plt.yticks(np.arange(0, 1.1, step=0.1)) 
    plt.tight_layout()
    plt.show()
    print(variances)

# Determine how much variance components in PCA explain
determine_n_PC(data=V_standardscaled)

#%% #! data=V [0.4759334, 0.7600839, 1.00234547,…]
pca = PCA(n_components=3)
pca.fit_transform(V)
# Get the eigenvectors (principal axes)
eigenvectors = pca.components_

# Get the eigenvalues (variances) of the principal components
eigenvalues = pca.explained_variance_

# Print the eigenvectors and eigenvalues
for i in range(len(eigenvalues)):
    print(f"Principal Component {i+1}:")
    print("Eigenvalue:", eigenvalues[i])
    print("Eigenvector:", eigenvectors[i])
    print()

# Transform the data using the principal components
transformed_data = pca.transform(V)
# Print the shape of transformed data (3276800, 2)
print("Transformed data shape:", transformed_data.shape)

Vmean = sum(DVFs_columns) / len(DVFs_columns)             
x = np.random.normal(0, 1, Vmean.shape)
#%%
print(eigenvectors.shape, eigenvalues.shape, x.shape) #(3, 27) (3,) (3276800, 3)
# (3, 27) (3,) -> (3, 27)
PCA_values = eigenvectors*eigenvalues[:, np.newaxis]
# (3, 3276800) (3276800, 3) -> (3,3)
temp = np.dot(PCA_values, x)



# (2,27)*(2,)*(3276800, 3)=(2,27)
PCA_values = eigenvectors*eigenvalues[:, np.newaxis]*x
# Idea1: (3276800, 3) + (2,27)
# vg1 = Vmean + PCA_values
# Idea2: (3276800, 27) * (2,27)
vg2 = V * PCA_values[0] + V * PCA_values[1]
Vmean = sum(DVFs_columns) / len(DVFs_columns)   

restored_array = np.reshape(vg2, (160, 128, 160, 3))





# assume PCA_values scale V

#%% #!data=V.T [0.4557183, 0.71959734, 0.9122899,...]
pca = PCA(n_components=3)
pca.fit_transform(V.T)
# Get the eigenvectors (principal axes)
eigenvectors = pca.components_

# Get the eigenvalues (variances) of the principal components
eigenvalues = pca.explained_variance_

# Print the eigenvectors and eigenvalues
for i in range(len(eigenvalues)):
    print(f"Principal Component {i+1}:")
    print("Eigenvalue:", eigenvalues[i])
    print("Eigenvector:", eigenvectors[i])
    print()

Vmean = sum(DVFs_columns) / len(DVFs_columns)             
x = np.random.normal(0, 1, Vmean.shape)
#%% 
print(eigenvectors.shape, eigenvalues.shape, x.shape) #!eigenvalues large
# (3, 3276800) (3,) -> (3, 3276800)
PCA_values = eigenvectors*eigenvalues[:, np.newaxis]/5000
# (3, 3276800) (3276800, 3) -> (3,3)
temp = PCA_values * np.random.normal(0, 1, (3,))[:, np.newaxis]
Vmethod1 = Vmean + temp.T
# -> (3276800, 3)


PCA_values = eigenvectors
# (3, 3276800) (3276800, 3) -> (3,3)
temp = PCA_values *5000
Vmethod2 = Vmean + temp.T

#%% method3: apply pca on only 1 DVF and add this to the vmean
# V=DVFs_columns[0]


#%% method4: GPT
# Step 1: Calculate the mean velocity field
# Load your velocity fields data into a numpy array
# velocity_fields = np.random.rand(3276800, 3)
Vone = DVFs_columns[0]
# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(Vone)

# Generate random velocity fields
V_prime = np.mean(Vone, axis=0)
U = pca.components_
d = pca.explained_variance_
x = np.random.normal(0, 1, (9, 2))
vg = V_prime + U.T @ (d * x.T)

#%%
"""
V               (3276800,27)
pca.fit(V.T) or pca.fit(V)
eigenvectors    (2,27)  or (2,3276800)
eigenvalues     (2)
x               (3276800, 3)

Vmean   (3276800,3) or (3276800,)    
    
    (3276800,3) + 
    
"""
#%% OPERATION IS V IN ONE DIRECTION
DVFs_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]  

new_DVFs = []
for j in range(4):
    # new_DVF = np.array([])
    new_DVF = np.zeros((DVFs_columns[0].shape[0], 3))
    for i in range(3):
        # Select OndeDirection (x, y, or z) of displacement fields
        DVFs_OD = [DVF[:,i] for DVF in DVFs_columns]
        # Scale data
        # standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
        # Vxs = standard_scalar.fit_transform(Vxs)
        
        # Fit PCA
        pca = PCA()
        pca.fit(DVFs_OD)
        # Determine the number of principal components for 90% variability
        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance_ratio_cumsum >= 0.9) + 1
        # print(num_components, explained_variance_ratio_cumsum)

        # Calculate variables
        U = pca.components_[:num_components,:]         #(2, 3276800)
        d = pca.explained_variance_[:num_components] #(2,)
        x = np.random.normal(loc=0, scale=10, size=num_components)
        DVF_OD_mean = np.mean(DVFs_OD, axis=0)

        # Compute new DVF_OD
        # Scale eigenvalues with dimensions, and multiply with eigenvectors to scale them
        U_scaled = U * (d[:, np.newaxis]/DVF_OD_mean.shape[0])   #(2, 3276800)   [ 0.0188702 ,  0.01868925   [-0.00175282, -0.00176007,
        # Add random array
        U_sr = np.dot(x, U_scaled)             #(3276800,)     [ 0.00820394,  0.00813201,  0.00806008
        # Add to average Vx 
        DVF_OD_new = DVF_OD_mean + U_sr
        
        # Adding direction to new DVF
        new_DVF[:,i]=DVF_OD_new
  
    # plt.imshow(np.reshape(new_DVF, (160, 128, 160, 3))[:,64,:,0])
    new_DVFs.append(np.reshape(new_DVF, (160, 128, 160, 3)))

#%%



#%% Transform data
img_moving = img_data_T00[5]

# Compare with mean DVF
DVF_mean = sum(DVFs_arrays) / len(DVFs_arrays)
transform(DVF_itk=DVF_mean, img_moving=img_moving)


for i in range(len(new_DVFs)):
    transform(DVF_itk=new_DVFs[i], img_moving=img_moving)
#%% Restore array
DVF_mean = sum(DVFs_arrays) / len(DVFs_arrays)

# Convert and DVFs take Vmean and reshape to og
DVFs_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]    
Vmean = sum(DVFs_columns) / len(DVFs_columns) 
DVF_vmean = np.reshape(Vmean, (160, 128, 160, 3))

# From PCA method1 -> values way to large
DVF_method1 = np.reshape(Vmethod1, (160, 128, 160, 3))
DVF_method2 = np.reshape(Vmethod2, (160, 128, 160, 3))

# Perform transformation


img_moving = img_grid
img_moving = img_data_T00[5]

# transform(DVF_itk=DVF_mean, img_moving=img_data_T00[imgnr])
# transform(DVF_itk=DVF_vmean, img_moving=img_data_T00[imgnr])
# transform(DVF_itk=DVF_method1, img_moving=img_data_T00[imgnr])
# transform(DVF_itk=DVF_method1, img_moving=img_data_T50[imgnr])
# transform(DVF_itk=DVF_method1, img_moving=img_moving)
# transform(DVF_itk=DVF_method2, img_moving=img_moving)
for i in range(new_DVFs):
    transform(DVF_itk=new_DVFs[i], img_moving=img_moving)
# %%
