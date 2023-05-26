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
from glob import glob
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

# sys.path.append(os.path.join(os.path.dirname(__file__),'data\\datasets.py'))
# from data import datasets 
import datasets 

#%% Data preparation
root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
train_dataset = datasets.DatasetLung('train', root_data=root_data, augment_def=False, phases="in_ex")

img_data_T00, img_data_T50, img_data_T90 = [], [], []
for i in range(len(train_dataset)):
    img_fixed, img_moving,_,_ = train_dataset[i]
    if i%2==0:
        img_data_T00.append(img_fixed.squeeze(0).numpy()), img_data_T50.append(img_moving.squeeze(0).numpy())  #fixed T50 are dubble, moving T00 and T90       
    else:           
        img_data_T90.append(img_fixed.squeeze(0).numpy())


# %% Functions
def registration(fixed_image, moving_image, method="rigid", plot=False, parameter_path=""):
    """Function that calls Elastix registration function"""
    # print("Registration step start ")

    # Define parameter object
    parameter_object = itk.ParameterObject.New()
    if "Par" in method:
        parameter_object.AddParameterFile(parameter_path+method)#e.g. Par007
        # parameter_path =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
        # method = "Par0007.txt"
    else:
        parameter_map = parameter_object.GetDefaultParameterMap(method)  # rigid
        # parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
        # parameter_map['NumberOfSpatialSamples'] = ['3000']
        parameter_object.AddParameterMap(parameter_map)
        # print(parameter_object)

    # Registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=True)

    # Deformation field
    deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
    print("Registration step complete")

    # Jacobanian
    jacobians = itk.transformix_jacobian(moving_image, result_transform_parameters)
    # Casting tuple to two numpy matrices for further calculations.
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float32)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float32)
    print("Number of foldings in transformation:",np.sum(det_spatial_jacobian < 0))
    
    if plot:
        plot4(fixed_image, moving_image, result_image, deformation_field)

    return result_image, deformation_field, result_transform_parameters

def plot4(fixed_image, moving_image, result_image, deformation_field):
    """Plot fixed, moving result image and deformation field
       Called after registration to see result"""
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(fixed_image[:, 64, :], cmap='gray')
    axs[0, 0].set_title("fixed_image")
    axs[0, 1].imshow(moving_image[:, 64, :], cmap='gray')
    axs[0, 1].set_title("moving_image")
    axs[1, 0].imshow(result_image[:, 64, :], cmap='gray')
    axs[1, 0].set_title("result_image")
    axs[1, 1].imshow(deformation_field[:, 64, :, 2], cmap='gray')
    axs[1, 1].set_title("deformation_field")
    plt.tight_layout()
    plt.show()

def generate_atlas(img_data):
    # Initialize atlas A0 - Select random train image
    random_image = random.randint(0, len(img_data)-1)
    print("1. Initializing atlas A0 as image {}".format(random_image))
    A0 = itk.image_from_array(img_data[random_image])
    
    # Rigidly register each image to A0 to obtain registered set IT0
    print("2. Registration of initial image set IT0 to A0 to acquire A1")
    registered_set = []  # IT0
    for img in img_data:
        moving_image = itk.image_from_array(img)
        
        result_image, deformation_field, _ = registration(
            fixed_image=A0, moving_image=moving_image, 
            method="rigid", plot=False)
        registered_set.append(result_image)

    # Obtain A1 by averaging registerd set IT0
    A1_array = sum(np.asarray(registered_set)) / len(np.asarray(registered_set)) 
    A1 = itk.image_from_array(A1_array.astype(np.float32))

    # Start iterative prcess to obtain final Atlas image A 
    print("3. Start iterative prcess to obtain final Atlas")
    A_new = A1
    iteration = 1
    max_iter = 50
    L2 = 1e6
    while iteration <= max_iter:
        An, L2_last, re_registered_set = A_new, L2, []
        # registered_set = re_registered_set #? Make new IT0 iteration

        # Registration 
        for i in range(len(registered_set)):
            print("Iteration {} image {}/{}".format(iteration, i+1, len(registered_set)))
            # Perform registration
            
            result_image, deformation_field, _ = registration(
                fixed_image=An, moving_image=registered_set[i], 
                method="affine", plot=False)  # ?Affine
            re_registered_set.append(result_image)

        A_new_array = sum(np.asarray(re_registered_set)) / len(np.asarray(re_registered_set)) 
        A_new_array = itk.image_from_array(A_new_array.astype(np.float32))
        
        plot4(An, A_new, np.asarray(An) - np.asarray(A_new), deformation_field)
        
        iteration += 1

        # Calculate L2 norm and break when convergence is reached
        L2 = np.linalg.norm(np.asarray(An) - np.asarray(A_new)) 
        print("L2: {}   dL2: {}".format(L2, L2_last - L2))
        
        if (L2_last - L2) <= 1:
            break
    
    print("Atlas generation complete")
    return An

#%% STEP 1 ATLAS GENERATION
# moving_image = itk.imread(data_path.format("01"))
# fixed_image = itk.imread(data_path.format("22"))
# parameter_path =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
# parameter_file = "Par0007.txt"
# parameter_file = "Par0011.txt"
# parameter_file = "Par0049.txt"
# parameter_file = "rigid"
# result_image, deformation_field = registration(
#             fixed_image=fixed_image, moving_image=moving_image, 
#             method=parameter_file, plot=True, parameter_path=parameter_path)

A = generate_atlas(img_data=img_data_T00)



#%% STEP 2 DVF GENERATION
def DVFs(img_data, img_atlas):
    """Generate displacement from set images registered on atlas image"""
    DVFs_list = []
    
    #! Now using deformation_field but in paper velocity field
    for i in range(len(img_data)):
        moving_image = itk.image_from_array(img_data[i])
        result_image, DVF, result_transform_parameters = registration(
                    fixed_image=img_atlas, moving_image=moving_image, 
                    method="affine", plot=True)  # ?Affine or  bspline
        
        DVFs_list.append(DVF)
    
    return DVFs_list

def apply_PCA(DVFs_list):
    #TODO apply PCA
    #-Organize DVFs in colmns of matrix and apply PCA
    #-Cut on 90% variability capture
    return DVFs_list 

def sample_DVFs(DVFs_list):
    #TODO
    #v = Vmean + U*x*d
    # return DVFs_list 
    
    
    # Take average of 
    DVF_summed = np.empty(DVFs_list[0].shape)
    for DVF in DVFs_list:
        DVF_summed += np.asarray(DVF)
    DVF_averaged = DVF_summed/len(DVFs_list)

    plot=True
    if plot:
        plt.imshow(DVF_averaged[:, 64, :, 2], cmap='gray')
    
    return DVF_averaged

#%%

DVFs = DVFs(img_data=img_data_T00[:4], img_atlas=A)

#%%
# plt.imshow(DVF[:, 64, :, 2], cmap='gray')

DVFs_arrays = [np.reshape(np.asarray(DVF), (-1, 3)) for DVF in DVFs]    #Turn each DVF into an array and reshapeto 3276800x3
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











# %% #! Validate atlas
img_data = img_data_T00[0:4]
DVFs_params = []
for i in range(len(img_data)):
    moving_image = itk.image_from_array(img_data[i])
    result_image, DVF, result_transform_parameters = registration(
                fixed_image=A, moving_image=moving_image, 
                method="affine", plot=True)  # ?Affine or  bspline
    
    DVFs_params.append(result_transform_parameters)

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





#%%
import gryds

reshaped_DVF = np.transpose(np.asarray(DVF), (3, 0, 1, 2)) #! gaat wss hier fout
plt.imshow(DVF[:,64,:,2], cmap='gray')
# plt.imshow(reshaped_DVF[2,:,64,:], cmap='gray')
normalized_array = reshaped_DVF / np.linalg.norm(reshaped_DVF, axis=0)

a_bspline_transformation = gryds.BSplineTransformation([reshaped_DVF[0], reshaped_DVF[1], reshaped_DVF[2]])

an_image_interpolator = gryds.Interpolator(moving_image)
# plt.imshow(an_image_interpolator[:,64,:])

a_deformed_image = an_image_interpolator.transform(a_bspline_transformation)
# plt.imshow(a_deformed_image, cmap='gray')
plt.imshow(a_deformed_image[:,64,:], cmap='gray')






#%%
an_image_interpolator = gryds.Interpolator(moving_image[:,64,:])

a_translation = gryds.TranslationTransformation([0, 0.1])
a_translated_image = an_image_interpolator.transform(a_translation)
plt.imshow(a_translated_image)


#%%
a_grid = gryds.Grid(moving_image.shape)
transformed_grid = a_grid.transform(a_bspline_transformation)
displacement_field = transformed_grid.grid - a_grid.grid

fig, ax = plt.subplots(1, 2)
ax[0].imshow(**gryds.dvf_show(displacement_field[0,:,64,:]));
ax[1].imshow(**gryds.dvf_show(displacement_field[1,:,64,:]));
ax[0].set_title('$u_x$');
ax[1].set_title('$u_y$');
#%%
moving_image = itk.image_from_array(img_data_T00[0])
result_image, DVF, result_transform_parameters = registration(
            fixed_image=A, moving_image=moving_image, 
            method="affine", plot=True)  # ?Affine or  bspline

moving_image = itk.image_from_array(img_data_T00[0])

# result_image_transformix = itk.transformix_filter(
#     moving_image,
#     result_transform_parameters)


plt.imshow(moving_image[:,64,:], cmap='gray')



plt.imshow(warped_image[:,64,:], cmap='gray')




# %%
import itk
import argparse

parser = argparse.ArgumentParser(description="Wrap An Image Using A Deformation Field.")
parser.add_argument("moving_image")
parser.add_argument("DVF")
parser.add_argument("output_image")
args = parser.parse_args()

DisplacementFieldType = type(DVF)
ImageType = type(moving_image)

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.moving_image)

fieldReader = itk.ImageFileReader[DisplacementFieldType].New()
fieldReader.SetFileName(args.DVF)
fieldReader.Update()

deformationField = fieldReader.GetOutput()

warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()

interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

warpFilter.SetInterpolator(interpolator)

warpFilter.SetOutputSpacing(deformationField.GetSpacing())
warpFilter.SetOutputOrigin(deformationField.GetOrigin())
warpFilter.SetOutputDirection(deformationField.GetDirection())

warpFilter.SetDisplacementField(deformationField)

warpFilter.SetInput(reader.GetOutput())

writer = itk.ImageFileWriter[ImageType].New()
writer.SetInput(warpFilter.GetOutput())
writer.SetFileName(args.output_image)

writer.Update()

#%%
parameter_object = itk.ParameterObject.New()
parameter_object.AddParameterFile("parameters.0.txt")

elastix_filter = itk.ElastixRegistrationMethod.New(fixed_image=fixed_image, moving_image=moving_image)
elastix_filter.SetParameterObject(parameter_object)
elastix_filter.Execute()

result_image = elastix_filter.GetResultImage()

displacement_field = elastix_filter.GetTransformParameterObject().GetDisplacementField()

resampler = itk.ResampleImageFilter.New(Input=result_image, Transform=itk.DisplacementFieldTransform[itk.D, 2].New(DVF), UseReferenceImage=True)
resampler.SetReferenceImage(moving_image)
resampler.Update()

warped_image = resampler.GetOutput()

# %%
