#%%
import itk
# Import the required libraries
import elastix
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
# import seaborn as sns
import numpy as np
# import SimpleITK as sitk
import scipy.stats as stats
# import openpyxl
import nibabel as nib


#%%
# Define the file path
fixed_image_path = '4DCT/data/train/image/case_001/T00.nii.gz'
moving_image_path = '4DCT/data/train/image/case_001/T50.nii.gz'
# Read the image using ITK
fixed_image = itk.imread(fixed_image_path)
moving_image = itk.imread(moving_image_path)


# Import Default Parameter Map
parameter_object = itk.ParameterObject.New()
# Rigid
# parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
# parameter_object.AddParameterMap(parameter_map_rigid)
# Bspline
# parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
# parameter_object.AddParameterMap(parameter_map_bspline)
# 
# parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0007.txt')
# parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0011.txt')
parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0049.txt')


# Call registration function and specify output directory
result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object,
    output_directory='output/')

deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)


#%%
def plot4(fixed_image, moving_image, result_image, deformation_field):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(fixed_image[:,64,:], cmap='gray')
    axs[0,0].set_title("fixed_image")
    axs[0,1].imshow(moving_image[:,64,:], cmap='gray')
    axs[0,1].set_title("moving_image")
    axs[1,0].imshow(result_image[:,64,:], cmap='gray')
    axs[1,0].set_title("result_image")
    axs[1,1].imshow(deformation_field[:,64,:,2], cmap='gray')
    axs[1,1].set_title("deformation_field")
    plt.tight_layout()
    plt.show()

# %%
### Adjust the following parameters ###

# path to the end-expiration phase
fixed_image_path = "4DCT/data/train/image/case_0{}/T00.nii.gz"

# path to the end-inspiration phase
moving_image_path = "4DCT/data/train/image/case_0{}/T50.nii.gz"

# parameter files evaluated
params = ['Par0007', 'Par0011', 'Par0049']

# cases of the data evaluated
cases = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# Perform registration & transformation or only visualization
registration_transformation = False

plot=True
for case in cases:
    for param in params:
        print("case:", case, "param:", param)
        fixed_image = itk.imread(fixed_image_path.format(case))
        moving_image = itk.imread(moving_image_path.format(case))
        
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/{}.txt'.format(param))
        
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object)

        deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
        print("Registration done")
        if plot:
            plot4(fixed_image, moving_image, result_image, deformation_field)


# %%
