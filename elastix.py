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
moving_image = itk.imread(moving_image_path, itk.F)


# Import Default Parameter Map
parameter_object = itk.ParameterObject.New()
parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')

parameter_object.AddParameterMap(parameter_map_rigid)
parameter_object.AddParameterMap(parameter_map_bspline)

# Call registration function and specify output directory
result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object)

deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)


#%%

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
#%%
ELASTIX_PATH = r"C:\Users\20203531\OneDrive - TU Eindhoven\Y3\Q4\BEP\Elixtar\elastix.exe"

param_file_path = os.path.join(r"path\Parameter files", param + ".txt")
result_path = os.path.join(path, param)
transform_file_path = os.path.join(path, param, "TransformParameters.0.txt")
output_dir_path = os.path.join(path, param)

el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[param_file_path],
    output_dir=result_path) 