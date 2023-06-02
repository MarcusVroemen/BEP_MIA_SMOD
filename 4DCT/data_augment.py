# %%
import itk
# Import the required libraries
# import elastix
# import imageio
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
# import seaborn as sns
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

#%% Generate atlas DVFs
DVFs = register_to_atlas(img_data=img_data_T00, img_atlas=img_atlas)
DVFs_arrays = [np.asarray(DVF) for DVF in DVFs]

#%% apply DVFmean to images
DVF_mean = sum(DVFs_arrays) / len(DVFs_arrays)
DVF_gryds = DVF_conversion(DVF_mean, plot=True)

for img_moving in img_data_T00:
    a_bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
    an_image_interpolator = gryds.Interpolator(img_moving)
    a_deformed_image = an_image_interpolator.transform(a_bspline_transformation)
    EF.plot_registration(fixed_image=img_moving, moving_image=a_deformed_image, deformation_field=DVF_mean, full=True,
                         result_image=np.asarray(img_moving) - np.asarray(a_deformed_image), title="transformation with averaged atlas DVF",
                         name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")

#%% #!--------PCA-----------
DVFs_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]    #Turn each DVF into an array and reshapeto 3276800x3
V = np.concatenate(DVFs_arrays, axis=1)                 #Concatenate DVFs into matrix V with shape 3276800x(3*len(DVFs_arrays))
Vmean = sum(DVFs_arrays) / len(DVFs_arrays)             #Average DVFs into matrix Vmean with shape 3276800x3       

# U eigenvectors of principal modes
# d eigenvalues of principal modes
# x array of random numbers following distribution N(0,simga)
# vg = Vmean + U*x*d

#%%#TODO Apply PCA ---------------------------------------------
pca = PCA(n_components=V.shape[1])
pca.fit(V)

Vtransform=pca.transform(V)
principal_components = pca.components_ #rows=principal components columns=Features

#plot components
plt.figure(figsize=(10,10))
plt.scatter(Vtransform[:,0],Vtransform[:,1])
plt.xlabel('pc1')
plt.ylabel('pc2')

# check how much variance is explained by each principal component
print(pca.explained_variance_ratio_)

eigenvalues = pca.explained_variance_
plt.plot(eigenvalues)
# Get the principal components and explained variance ratios
explained_variance_ratios = pca.explained_variance_ratio_
plt.plot(explained_variance_ratios)

# Transform the data using the principal components
transformed_data = pca.transform(V)

# Print the shape of transformed data
print("Transformed data shape:", transformed_data.shape)



# %%
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000)
n_samples = X.shape[0]

pca2 = PCA()
X_transformed = pca2.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca2.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca2.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)



#%%
#! -------------PCA-----------------
img_data = img_data_T00[0:4]
DVFs = []
for i in range(len(img_data)):
    moving_image = itk.image_from_array(img_data[i])
    result_image, DVF, result_transform_parameters = registration(
                fixed_image=A, moving_image=moving_image, 
                method="affine", plot=True)  # ?Affine or  bspline
    DVFs.append(DVF)
DVFs_arrays = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]    #Turn each DVF into an array and reshapeto 3276800x3
V = np.concatenate(DVFs_arrays, axis=1)                 #Concatenate DVFs into matrix V with shape 3276800x(3*len(DVFs_arrays))
Vmean = sum(DVFs_arrays) / len(DVFs_arrays) 
DVF = DVFs_arrays[0]

#%% Scale data
from sklearn import preprocessing
standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
DVF_scaled = standard_scalar.fit_transform(DVF)
import seaborn as sns
sns.boxplot(data=DVF_scaled)

#%% PCA



pca = PCA(n_components=DVF.shape[1])
pca.fit(DVF)

transform=pca.transform(DVF)
principal_components = pca.components_ #rows=principal components columns=Features

# check how much variance is explained by each principal component
print(pca.explained_variance_ratio_)

eigenvalues = pca.explained_variance_
plt.plot(eigenvalues)
eigenvector = pca.components_

# Get the principal components and explained variance ratios
explained_variance_ratios = pca.explained_variance_ratio_
plt.plot(explained_variance_ratios)

# Transform the data using the principal components
transformed_data = pca.transform(DVF)

# Print the shape of transformed data
print("Transformed data shape:", transformed_data.shape)

DVF_new = np.reshape(transformed_data, DVFs[0].shape)

input_image = itk.image_from_array(img_data_T00[0])
deform_image(input_image, DVFs[0], plot=True)
deform_image(input_image, DVF_new, plot=True) #same type error itkImageVF33




#%% 





moving_image = np.asarray(img_data_T00[0])
DVF=np.asarray(DVFs[0])

transform = itk.TranslationTransform.New()
transform.SetOffset(DVF)

number_of_columns = 5
number_of_rows = 6

moving_image = ImageType.New()
moving_image.SetRegions([number_of_columns, number_of_rows])
moving_image.Allocate(True)

# Set the pixel values consecutively to 1, 2, 3, ..., n.
moving_image[:] =  np.arange(1, number_of_columns*number_of_rows + 1).reshape((number_of_rows, number_of_columns))
                            
print('Moving image:')
print(np.asarray(moving_image))
print()

translation = [1, -2]
print('Translation:', translation)

transform = itk.TranslationTransform.New()
transform.SetOffset(translation)

parameter_map = {
                 "Direction": ("1", "0", "0", "1"),
                 "Index": ("0", "0"),
                 "Origin": ("0", "0"),
                 "Size": (str(number_of_columns), str(number_of_rows)),
                 "Spacing": ("1", "1")
                }

parameter_object = itk.ParameterObject.New()
parameter_object.AddParameterMap(parameter_map)

transformix_filter.SetMovingImage(moving_image)
transformix_filter.SetTransformParameterObject(parameter_object)
transformix_filter.SetTransform(transform)
transformix_filter.Update()

output_image = transformix_filter.GetOutput()

print('Output image:')
print(np.asarray(output_image))
print()