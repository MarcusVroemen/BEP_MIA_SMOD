# %%
import itk
# Import the required libraries
# import elastix
# import imageio
import matplotlib.pyplot as plt
# from matplotlib import colors, rcParams
import seaborn as sns
# import scipy.stats as stats
# import openpyxl
import nibabel as nib
# import random
import os
# import sys
from sklearn.decomposition import PCA
# import torch
# import torch.nn.functional as F
# import torch.utils.data
from matplotlib import pyplot as plt
# from tqdm import tqdm
import numpy as np
# import SimpleITK as sitk
# import cv2
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
            
# validateDVFT00s(img_data=img_data_T00, img_atlas=img_atlas, method="affine")
# conclusion: affine gives ok results


#%% STEP 2 DVF GENERATION #################################################################
def register_to_atlas(img_data, img_atlas):
    """Generate DVFs from set images registered on atlas image"""
    params_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/transform_parameters/"
    DVFs_list, DVFs_inverse_list, result_images = [], [], []
    result_images = []
    for i in range(len(img_data)):
        result_image, DVF, result_transform_parameters = EF.registration(
            fixed_image=img_atlas, moving_image=img_data[i], 
            method="affine", plot=True, 
            output_directory=params_path) 
        print("start inverse")
        # Inverse DVF
        parameter_object = itk.ParameterObject.New()
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
        parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
        parameter_map_bspline['HowToCombineTransforms'] = ['Compose']
        parameter_map_rigid['HowToCombineTransforms'] = ['Compose']
        parameter_object.AddParameterMap(parameter_map_rigid)
        parameter_object.AddParameterMap(parameter_map_bspline)
        
        inverse_image, inverse_transform_parameters = itk.elastix_registration_method(
            img_data[i], img_data[i],
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=params_path+"TransformParameters.0.txt")
        
        inverse_transform_parameters.SetParameter(
            0, "InitialTransformParametersFileName", "NoInitialTransform")
        
        DVF_inverse = itk.transformix_deformation_field(img_data[i], inverse_transform_parameters)
        
        DVFs_list.append(DVF)
        DVFs_inverse_list.append(DVF_inverse)
        result_images.append(result_image)
        
    return DVFs_list, DVFs_inverse_list, result_images

def DVF_conversion(DVF_itk): 
    """Converts DVF from itk to DVF usable with gryds"""
    # Reshapre DVF from (160, 128, 160, 3) to (3, 160, 128, 160)
    reshaped_DVF = np.transpose(np.asarray(DVF_itk), (3, 0, 1, 2))  
    
    # DVF elastix in pixels while DVF gryds in proportions. Therefore scale each direction with their pixelsize. 
    # DVF[0] works medial-lateral on 160 pixels, DVF[2] cranial-caudial on 160 pixels, DVF[1] posterior-anterior in 128 pixels
    DVF_scaled = np.asarray([reshaped_DVF[0]/160, reshaped_DVF[1]/128, reshaped_DVF[2]/160])
    
    # gryds transformation takes DVFs in different order: CC, AP, ML
    DVF_gryds = [DVF_scaled[2], DVF_scaled[1], DVF_scaled[0]]

    return DVF_gryds

def transform(DVF_itk, img_moving, plot=False):
    DVF_gryds = DVF_conversion(DVF_itk)
    # DVF_gryds = DVF_itk
    bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
    an_image_interpolator = gryds.Interpolator(img_moving)
    img_deformed = an_image_interpolator.transform(bspline_transformation)
    if plot:
        EF.plot_registration(fixed_image=img_moving, moving_image=img_deformed, deformation_field=DVF_itk, full=True, 
                            result_image=np.asarray(img_moving) - np.asarray(img_deformed), title="transformation with averaged atlas DVF",
                            name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")
    return img_deformed

def dimreduction(DVFs):
    """Apply PCA to get DVF values for expression of DVF generation
    Args:
        DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
    Returns:
        DVF_mean (list with arrays): For each direction the mean of inputted DVFs
        DVF_Ud (list with arrays): For each direction the PCA components needed to generate the artificial DVFs
    """
    # Convert (160, 128, 160, 3) to (3276800, 3)
    DVFs_columns = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]  
    # Make empty lists for args to return
    DVF_mean, DVF_Ud = [],[]
    
    for i in range(3):
        # Select OndeDirection (x, y, or z) of displacement fields
        DVFs_OD = [DVF[:,i] for DVF in DVFs_columns]
        # Average DVFs from one direction and add to list to return
        DVF_OD_mean = np.mean(DVFs_OD, axis=0) 
        DVF_mean.append(DVF_OD_mean)
        
        # Scale data
        standard_scalar = preprocessing.StandardScaler(with_mean=True, with_std=True)
        DVFs_OD = standard_scalar.fit_transform(DVFs_OD)
        
        # Fit PCA
        pca = PCA()
        pca.fit(DVFs_OD)
        # Determine the number of principal components for 90% variability
        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance_ratio_cumsum >= 0.9) + 1
        # print(num_components, explained_variance_ratio_cumsum)

        # Calculate PCA variables to return
        U = pca.components_[:num_components,:]              #(2, 3276800)
        d = pca.explained_variance_[:num_components]        #(2,)
        # Scale eigenvectors with eigenvalues, that are scaled with the total amount of variance
        d_scaled = d/sum(pca.explained_variance_)           #sum_all=1
        Ud = U * d_scaled[:, np.newaxis]                    #(p, 3276800)  
        # Adding U*d to list with 3 dimensions
        DVF_Ud.append(Ud)

    return DVF_mean, DVF_Ud
  
def generate_artificial_DVFs(amount, sigma, DVFs):
    """Generate artificial DVFs from list with to_atlas_DVFs
    Args:
        amount (int): amount of artificial DVFs to generate
        sigma (int): random scaling component 100: visual deformations, 500: too much
        DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
    Returns:
        DVFs_artificial (list with arrays): artificial DVFs with shape (160, 128, 160, 3)
    """
    # Call dimreduction to calculate mean and PCA components
    DVF_mean, DVF_Ud = dimreduction(DVFs)
    
    DVFs_artificial = []
    for i in range(amount):
        DVF_artificial = np.zeros((DVF_mean[0].shape[0], 3))
        for j in range(3):
            x = np.random.normal(loc=0, scale=sigma, size=DVF_Ud[j].shape[0])
            # Paper: vg = Vmean + U*x*d 
            DVF_artificial[:,j] = DVF_mean[j] + np.dot(DVF_Ud[j].T, x)  #(3276800,1)
        DVFs_artificial.append(np.reshape(DVF_artificial, (160, 128, 160, 3)))

    return DVFs_artificial

#%% Generate atlas DVFs + inverse
DVFs, DVFs_inverse, T00_to_atlas = register_to_atlas(img_data=img_data_T00, img_atlas=img_atlas)

#%% VALIDATE INVERSE: For every DVF plot DVF, DVF_inverse and their effect on a training image
def validate_inverse(DVFs, DVFs_inverse, img_moving):
    fig, ax = plt.subplots(len(DVFs),4, figsize=(40,50))
    for i in range(len(DVFs)):
        ax[i,0].imshow(np.asarray(DVFs[i])[:,64,:,2])
        img_ogDVF = transform(DVF_itk=DVFs[i], img_moving=img_moving, plot=False)
        ax[i,1].imshow(img_ogDVF[:,64,:])
        ax[i,2].imshow(np.asarray(DVFs_inverse[i])[:,64,:,2])
        img_invDVF = transform(DVF_itk=DVFs_inverse[i], img_moving=img_moving, plot=False)
        ax[i,3].imshow(img_invDVF[:,64,:])
    plt.tight_layout()
    plt.show()
# validate_inverse(DVFs, DVFs_inverse, img_moving=img_data_T00[0])
#%%

#%% Generate artificial DVFs
# DVFs_artificial = generate_artificial_DVFs(amount=5, sigma=100,DVFs=DVFs)
DVFs_artificial_inverse100 = generate_artificial_DVFs(amount=5, sigma=100,DVFs=DVFs_inverse)
DVFs_artificial_inverse200 = generate_artificial_DVFs(amount=5, sigma=200,DVFs=DVFs_inverse)
DVFs_artificial_inverse300 = generate_artificial_DVFs(amount=5, sigma=300,DVFs=DVFs_inverse)
DVFs_artificial_inverse500 = generate_artificial_DVFs(amount=5, sigma=500,DVFs=DVFs_inverse)
#%%
# img_moving = img_data_T00[1]
# fig, ax = plt.subplots(len(DVFs_artificial),4, figsize=(40,50))
# for i in range(len(DVFs_artificial)):
#     ax[i,0].imshow(np.asarray(DVFs_artificial[i])[:,64,:,2])
#     img_ogDVF = transform(DVF_itk=DVFs_artificial[i], img_moving=img_moving, plot=False)
#     ax[i,1].imshow(img_ogDVF[:,64,:])
#     ax[i,2].imshow(np.asarray(DVFs_artificial_inverse[i])[:,64,:,2])
#     img_invDVF = transform(DVF_itk=DVFs_artificial_inverse[i], img_moving=img_moving, plot=False)
#     ax[i,3].imshow(img_invDVF[:,64,:])
# plt.tight_layout()
# plt.show()

#%% Generate arti
DVFs_artificial_inverse = DVFs_artificial_inverse500
for i in range(len(T00_to_atlas)):
    for j in range(len(DVFs_artificial_inverse[:3])):
        img_artificial = transform(DVF_itk=DVFs_artificial_inverse[j], img_moving=T00_to_atlas[i], plot=False)
        EF.plot_registration(fixed_image=img_data_T00[i], moving_image=T00_to_atlas[i], result_image=img_artificial, deformation_field=DVFs_artificial_inverse[j],
                             name1="T00 original", name2="T00 to atlas", name3="artificial T00", name4="artificial DVF",
                             title="Creating artificial images", full=True)
