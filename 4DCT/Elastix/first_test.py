#%%
import itk
# Import the required libraries
# import elastix
# import os
# import imageio
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams
# import seaborn as sns
import numpy as np
# import SimpleITK as sitk
# import scipy.stats as stats
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
base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/"

# path to the end-expiration phase
fixed_image_path = base_path+"data/train/image/case_0{}/T00.nii.gz"

# path to the end-inspiration phase
moving_image_path = base_path+"data/train/image/case_0{}/T50.nii.gz"

# parameter files evaluated
params = ['Par0007', 'Par0011', 'Par0049']

# cases of the data evaluated
cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"] #cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]


#%%
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
            parameter_object=parameter_object,
            log_to_console=True)

        deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
        print("Registration done")
        if plot:
            plot4(fixed_image, moving_image, result_image, deformation_field)


# %% Step 1.1
# Randomly select one T00 image as initial altas A0 
T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
import random
A0_case = cases[random.randint(0,len(cases)-1)]
A0_path = T00_path.format(A0_case)

DVFs = []
registered_images = []  #IT0
# Register every T00 to A0 and save DVFs
for case in cases:
    print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
    A0 = itk.imread(A0_path)
    moving_image = itk.imread(T00_path.format(case)) #Dont register A0 image?
    # print(A0.shape, moving_image.shape)
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)

    #! ITK ERROR in some cases
    result_image, result_transform_parameters = itk.elastix_registration_method(
        A0, moving_image,
        parameter_object=parameter_object,
        log_to_console=True)
    
    deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
    
    registered_images.append(result_image)
    DVFs.append(deformation_field)
    
    print("Registration case 0{} done".format(case))
    
    plot=True
    if plot:
        plot4(A0, moving_image, result_image, deformation_field)
        
# %%
# Average registered images IT0 to create A1
A1_sum = np.empty(registered_images[0].shape)
for registered_image in registered_images:
    # print(np.asarray(registered_image)[0,0,0])
    A1_sum+=np.asarray(registered_image)
    # print(A1_sum[0,0,0])
A1=A1_sum/len(registered_images)
plt.imshow(A1[:,64,:], cmap='gray')
# %% #TODO
# Randomly select one T00 image as initial altas A0 
# T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
# import random
# A0_case = cases[random.randint(0,len(cases))]
# A0_path = T00_path.format(A0_case)

DVFs2 = []
registered_images2 = []  #IT0
# Register every T00 to A0 and save DVFs
for case in cases:
    print("Registering case 0{} to An".format(case))
    # fixed_image = itk.imread(A0_path)
    moving_image = itk.imread(T00_path.format(case)) #Dont register A0 image?
    # print(fixed_image.shape, moving_image.shape)
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)

    #! ITK ERROR in some cases
    result_image, result_transform_parameters = itk.elastix_registration_method(
        A1, moving_image,
        parameter_object=parameter_object,
        log_to_console=True)
    
    deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
    
    registered_images2.append(result_image)
    DVFs.append(deformation_field)
    
    print("Registration case 0{} done".format(case))
    
    plot=True
    if plot:
        plot4(fixed_image, moving_image, result_image, deformation_field)
# %%
