# %%
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
import random

# %%
### Adjust the following parameters ###
base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/"

# path to the end-expiration phase
fixed_image_path = base_path+"train/image/case_0{}/T00.nii.gz"

# path to the end-inspiration phase
moving_image_path = base_path+"train/image/case_0{}/T50.nii.gz"

# parameter files evaluated
params = ['Par0007', 'Par0011', 'Par0049']

# cases of the data evaluated
# cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]
cases = ["01", "02", "05", "06", "08"]


"""---------------------------------------------------------------------------------------------
- cases 22+ other dimensions
- from itk image types to np arrays
---------------------------------------------------------------------------------------------"""


# %%
def plot4(fixed_image, moving_image, result_image, deformation_field):
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


def register(fixed_image, moving_image, method="rigid", plot=False):
    print("Registration start ")

    # Define parameter object
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap(
        method)  # rigid
    parameter_object.AddParameterMap(parameter_map_rigid)

    # Registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        A0, moving_image,
        parameter_object=parameter_object,
        log_to_console=True)

    # Deformation field
    deformation_field = itk.transformix_deformation_field(
        moving_image, result_transform_parameters)
    print("Registration complete")

    if plot:
        plot4(fixed_image, moving_image, result_image, deformation_field)

    return result_image, deformation_field


def average_images(images_list, plot=False):
    """images_list with tensors either registered images or DVFs#!"""
    images_summed = np.empty(images_list[0].shape)
    for registered_image in images_list:
        # print(np.asarray(registered_image)[0,0,0])
        images_summed += np.asarray(registered_image)
        # print(images_summed[0,0,0])
    images_averaged = images_summed/len(images_list)

    if plot:
        plt.imshow(images_averaged[:, 64, :], cmap='gray')

    return images_averaged


def atlas_iterate(max_iter, A1, registered_set):
    A_current = A1
    iteration = 1
    convergence = []
    while iteration <= max_iter:
        re_registered_set = []
        print("Registering IT0 to A for new iteration of A, iteration: ", iteration)
        for i in range(len(registered_set)):
            print("Iteration {}/{} image {}/{}".format(iteration,
                  max_iter, i, len(registered_set)))
            # print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
            result_image, deformation_field = register(
                A_current, registered_set[i], method="affine", plot=True)  # ?Affine
            re_registered_set.append(result_image)

        A_new = average_images(re_registered_set, plot=True)
        iteration += 1

        L2 = np.linalg.norm(A_current - A_new)  # L2 norm
        print(L2)
        convergence.append(L2)
        plt.imshow(A_current[:, 64, :], cmap='gray')
        plt.imshow(A_new[:, 64, :], cmap='gray')
        plt.imshow(np.absolute(A_current - A_new)[:, 64, :], cmap='gray')

        A_current = A_new

        # TODO add convergence break
    return A_current, convergence


def A0_to_A1(cases, data_path, respitory_phase):
    A0_case = cases[random.randint(0, len(cases)-1)]
    A0_path = data_path.format(A0_case, respitory_phase)
    A0 = itk.imread(A0_path)
    print()
    registered_set = []  # IT0
    for case in cases:
        moving_image = itk.imread(data_path.format(case))
        result_image, deformation_field = register(
            A0, moving_image, method="rigid", plot=False)
        registered_set.append(result_image)

    A1 = average_images(registered_set)

    return A1


# %%
base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/"
data_path = base_path+"train/image/case_0{}/T00.nii.gz"
cases = ["01", "02", "05", "06", "08"]
# cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]
cases = ["01", "02", "05"]
# A0_to_A1(cases=cases, data_path=data_path, respitory_phase="T00")

# Initialize atlas A0 - Select random train image
A0_case = cases[random.randint(0, len(cases)-1)]
A0_path = data_path.format(A0_case)
A0 = itk.imread(A0_path)

# Rigidly register each image to A0 to obtain registered set IT0
registered_set = []  # IT0
for case in cases:
    print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
    moving_image = itk.imread(data_path.format(case))
    result_image, deformation_field = register(
        A0, moving_image, method="rigid", plot=False)
    registered_set.append(result_image)

# Obtain A1 by averaging registerd set IT0
A1 = average_images(registered_set)


A, convergence = atlas_iterate(
    max_iter=4, A1=A1, registered_set=registered_set)

# Non-rigdly register IT0 to A1 and average to otain A2 #?Affine or bspline
# iteration=1
# max_iteration=10
# A=A1
# while iteration<=max_iteration:
#     reregistered_set = []
#     print("Registering IT0 to A for new iteration of A, iteration: ", iteration)
#     for i in range(len(registered_set)):
#         print("Iteration: {}/{}    Image: {}/{}".format(iteration, max_iteration, i, len(registered_set)))
#         # print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
#         result_image, deformation_field = register(A, registered_set[i], method="affine", plot=True)
#         reregistered_set.append(result_image)
#     A_new = average_images(reregistered_set, plot=True)
#     iteration+=1

#     # diff = np.absolute(A-A_new)
#     distance = np.linalg.norm(A - A_new) #L2 norm
#     print(distance)
#     plt.imshow(np.absolute(A-A_new)[:,64,:], cmap='gray')


# %%
# def reg():
#     plot=True
#     for case in cases:
#         for param in params:
#             print("case:", case, "param:", param)
#             fixed_image = itk.imread(fixed_image_path.format(case))
#             moving_image = itk.imread(moving_image_path.format(case))

#             parameter_object = itk.ParameterObject.New()
#             parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/{}.txt'.format(param))


#             result_image, result_transform_parameters = itk.elastix_registration_method(
#                 fixed_image, moving_image,
#                 parameter_object=parameter_object,
#                 log_to_console=True)

#             deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
#             print("Registration done")
#             if plot:
#                 plot4(fixed_image, moving_image, result_image, deformation_field)

# %% Step 1.1
# def atlas_generation(atlas="random"):
#     # Randomly select one T00 image as initial altas A0
#     base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/"
#     T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
#     cases = ["01", "02", "05", "06", "08"]
#     cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]


#     if atlas=="random":
#         A0_case = cases[random.randint(0,len(cases)-1)]
#         A0_path = T00_path.format(A0_case)
#         A0 = itk.imread(A0_path)
#     else:
#         A0_raw=atlas
#         A0 = itk.GetImageFromArray(A0_raw)
#         A0 = itk.GetImageFromArray(np.ascontiguousarray(A0_raw))
#         # A0 = itk.imread(A0_raw)
#         # A0 = itk.GetImageFromArray(A0_raw)
#         A0_case="A case"

#     DVFs = []
#     registered_images = []  #IT0
#     # Register every T00 to A0 and save DVFs
#     for case in cases:
#         print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
#         moving_image = itk.imread(T00_path.format(case)) #Dont register A0 image?
#         # print(A0.shape, moving_image.shape)

#         # Define parameter object
#         parameter_object = itk.ParameterObject.New()
#         # parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid') #rigid
#         # parameter_object.AddParameterMap(parameter_map_rigid)
#         parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0049.txt')
#         print(parameter_object)

#         print(type(A0), A0.shape, type(moving_image), moving_image.shape)
#         fig, axs = plt.subplots(1, 2)
#         axs[0].imshow(A0[:,64,:], cmap='gray')
#         axs[0].set_title("A0")
#         axs[1].imshow(moving_image[:,64,:], cmap='gray')
#         axs[1].set_title("moving_image")


#         #! ITK ERROR in some cases
#         # registration_method = itk.elastix_registration_method(A0, moving_image, parameter_object=parameter_object)
#         # registration_method[0].LogToConsoleOn()
#         # result_image, result_transform_parameters = registration_method.Execute()

#         result_image, result_transform_parameters = itk.elastix_registration_method(
#             A0, moving_image,
#             parameter_object=parameter_object,
#             log_to_console=True)


#         deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
#         registered_images.append(result_image)
#         DVFs.append(deformation_field)


#         # fig, axs = plt.subplots(1, 2)
#         # axs[0].imshow(result_image[:,64,:], cmap='gray')
#         # axs[0].set_title("result")
#         # axs[1].imshow(deformation_field[:,64,:], cmap='gray')
#         # axs[1].set_title("DVF")

#         am = np.asarray(A0)-np.asarray(moving_image)
#         ar = np.asarray(A0)-np.asarray(result_image)
#         fig, axs = plt.subplots(1, 2)
#         axs[0].imshow(am[:,64,:], cmap='gray')
#         axs[0].set_title("A0-moving_image")
#         axs[1].imshow(ar[:,64,:], cmap='gray')
#         axs[1].set_title("A0-result_image")


#         print("Registration case 0{} done".format(case))

#         plot=False
#         if plot:
#             plot4(A0, moving_image, result_image, deformation_field)

#     return registered_images

# def average_A0(registered_images):
#     A1_sum = np.empty(registered_images[0].shape)
#     for registered_image in registered_images:
#         # print(np.asarray(registered_image)[0,0,0])
#         A1_sum+=np.asarray(registered_image)
#         # print(A1_sum[0,0,0])
#     A1=A1_sum/len(registered_images)
#     plt.imshow(A1[:,64,:], cmap='gray')

#     return A1


# registered_images = atlas_generation(atlas="random")
# A1 = average_A0(registered_images)

# A0 = itk.image_from_array(A1.astype(np.float32))
# registered_images2 = atlas_generation(atlas=A0)
# # A2 = average_A0(registered_images2)

# %%


# %%
# base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/"
# T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
# cases = ["01", "02", "05", "06", "08"]

# A0_case = cases[random.randint(0,len(cases)-1)]
# A0_path = T00_path.format(A0_case)
# A = itk.imread(A0_path, itk.F)
# %%
# Average registered images IT0 to create A1

# ITK ERROR: ElastixRegistrationMethod(000001721B207010): Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn()).
# ITK ERROR: ElastixRegistrationMethod(000001721B202ED0): Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn()).


# %% #TODO
# Randomly select one T00 image as initial altas A0
# T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
# import random
# A0_case = cases[random.randint(0,len(cases))]
# A0_path = T00_path.format(A0_case)

# DVFs2 = []
# registered_images2 = []  #IT0
# # Register every T00 to A0 and save DVFs
# for case in cases:
#     print("Registering case 0{} to An".format(case))
#     # fixed_image = itk.imread(A0_path)
#     moving_image = itk.imread(T00_path.format(case)) #Dont register A0 image?
#     # print(fixed_image.shape, moving_image.shape)
#     parameter_object = itk.ParameterObject.New()
#     parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
#     parameter_object.AddParameterMap(parameter_map_rigid)

#     #! ITK ERROR in some cases
#     result_image, result_transform_parameters = itk.elastix_registration_method(
#         A1, moving_image,
#         parameter_object=parameter_object,
#         log_to_console=True)

#     deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)

#     registered_images2.append(result_image)
#     DVFs.append(deformation_field)

#     print("Registration case 0{} done".format(case))

#     plot=True
#     if plot:
#         plot4(fixed_image, moving_image, result_image, deformation_field)
# %%


# Randomly select one T00 image as initial altas A0
# atlas="random"
# base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/"
# T00_path = base_path+"data/train/image/case_0{}/T00.nii.gz"
# cases = ["01", "02", "05", "06", "08"]

# if atlas=="random":
#     A0_case = cases[random.randint(0,len(cases)-1)]
#     A0_path = T00_path.format(A0_case)
#     A0 = itk.imread(A0_path)
# else:
#     A0=atlas

# DVFs = []
# registered_images = []  #IT0
# # Register every T00 to A0 and save DVFs
# for case in cases:
#     print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
#     moving_image = itk.imread(T00_path.format(case)) #Dont register A0 image?
#     # print(A0.shape, moving_image.shape)

#     # Define parameter object
#     parameter_object = itk.ParameterObject.New()
#     parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid') #rigid
#     parameter_object.AddParameterMap(parameter_map_rigid)
#     # print(parameter_object)

#     print(type(A0), A0.shape, type(moving_image), moving_image.shape)
#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(A0[:,64,:], cmap='gray')
#     axs[0].set_title("A0")
#     axs[1].imshow(moving_image[:,64,:], cmap='gray')
#     axs[1].set_title("moving_image")

#     #! ITK ERROR in some cases
#     result_image, result_transform_parameters = itk.elastix_registration_method(
#         A0, moving_image,
#         parameter_object=parameter_object,
#         log_to_console=True)

#     deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)

#     registered_images.append(result_image)
#     DVFs.append(deformation_field)

#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(result_image[:,64,:], cmap='gray')
#     axs[0].set_title("result")
#     axs[1].imshow(deformation_field[:,64,:], cmap='gray')
#     axs[1].set_title("DVF")

#     print("Registration case 0{} done".format(case))

#     plot=False
#     if plot:
#         plot4(A0, moving_image, result_image, deformation_field)


# %%
# %%
# Define the file path
# fixed_image_path = '4DCT/data/train/image/case_001/T00.nii.gz'
# moving_image_path = '4DCT/data/train/image/case_001/T50.nii.gz'
# # Read the image using ITK
# fixed_image = itk.imread(fixed_image_path)
# moving_image = itk.imread(moving_image_path)


# # Import Default Parameter Map
# parameter_object = itk.ParameterObject.New()
# # Rigid
# # parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
# # parameter_object.AddParameterMap(parameter_map_rigid)
# # Bspline
# # parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
# # parameter_object.AddParameterMap(parameter_map_bspline)
# #
# # parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0007.txt')
# # parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0011.txt')
# parameter_object.AddParameterFile('C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/Par0049.txt')


# # Call registration function and specify output directory
# result_image, result_transform_parameters = itk.elastix_registration_method(
#     fixed_image, moving_image,
#     parameter_object=parameter_object,
#     output_directory='output/')

# deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
