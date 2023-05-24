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

from data import datasets

device = 'cpu'
root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
train_dataset = DatasetLung('train', root_data=root_data, augment_def=False)
a,b,c,d = train_dataset[1]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

for batch_idx, (img_moving, img_fixed, lbl_moving, lbl_fixed) in enumerate(tqdm(train_loader, file=sys.stdout)):
    # Take the img_moving and fixed images to the GPU
    img_moving, img_fixed = img_moving.to(device), img_fixed.to(device)
































# %%
def registration(fixed_image, moving_image, method="rigid", plot=False, parameter_path=""):
    """Function that calls Elastix registration function"""
    print("Registration start ")

    # Define parameter object
    parameter_object = itk.ParameterObject.New()
    if "Par" in method:
        parameter_object.AddParameterFile(parameter_path+method)#e.g. Par007
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
    print("Registration complete")

    if plot:
        plot4(fixed_image, moving_image, result_image, deformation_field)

    return result_image, deformation_field

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
    print("Start registration of An iteratively")
    # Set initial conditions
    An = A1
    iteration = 1
    convergence = []
    
    #TODO Iterate until convergence
    while iteration <= max_iter:
        re_registered_set = []
        # Registration 
        for i in range(len(registered_set)):
            print("Iteration {}/{} image {}/{}".format(iteration, max_iter, i+1, len(registered_set)))
            # Perform registration
            #! An = itk.image_from_array(An.astype(np.float32))
            
            fig, axs = plt.subplots(2)
            axs[0].imshow(An[:, 64, :], cmap='gray')
            axs[0].set_title("An")
            axs[1].imshow(registered_set[i][:, 64, :], cmap='gray')
            axs[1].set_title("moving_image")
            
            result_image, deformation_field = registration(
                fixed_image=An, moving_image=registered_set[i], 
                method="affine", plot=False)  # ?Affine
            re_registered_set.append(result_image)

        A_new = average_images(re_registered_set, plot=False)
        iteration += 1

        L2 = np.linalg.norm(An - A_new)  # L2 norm
        print("L2: ",L2)
        convergence.append(L2)
        
        plot4(An, A_new, np.absolute(An - A_new), deformation_field)
        

        An = A_new
        registered_set = re_registered_set #? Make new IT0 iteration
        
    return An, convergence

def generate_atlas(data_path, cases):
    # Initialize atlas A0 - Select random train image
    A0_case = cases[random.randint(0, len(cases)-1)]
    A0_path = data_path.format(A0_case)
    A0 = itk.imread(A0_path)

    # Rigidly register each image to A0 to obtain registered set IT0
    registered_set = []  # IT0
    for case in cases:
        print("Registering case 0{} to A0 case 0{}".format(case, A0_case))
        moving_image = itk.imread(data_path.format(case))
        result_image, deformation_field = registration(
            fixed_image=A0, moving_image=moving_image, 
            method="rigid", plot=False)
        registered_set.append(result_image)

    # Obtain A1 by averaging registerd set IT0
    A1 = average_images(registered_set)

    # Start iterative prcess to obtain final Atlas image A 
    A, convergence = atlas_iterate(max_iter=4, A1=A1, registered_set=registered_set)
    
    return A

# %%
base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/"
data_path = base_path+"train/image/case_0{}/T00.nii.gz"
cases = ["01", "02", "05", "06", "08"]
# cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]
# cases = ["01", "02", "05"]

A, _ = generate_atlas(data_path, cases)

#%%
moving_image = itk.imread(data_path.format("01"))
fixed_image = itk.imread(data_path.format("22"))
parameter_path =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
parameter_file = "Par0007.txt"
# parameter_file = "Par0011.txt"
# parameter_file = "Par0049.txt"
parameter_file = "rigid"
# result_image, deformation_field = registration(
#             fixed_image=fixed_image, moving_image=moving_image, 
#             method=parameter_file, plot=True, parameter_path=parameter_path)









# %% DVFs step 2
def DVFs(atlas_image, data_path, cases):
    DVFs_list = []
    for case in cases:
        moving_image = itk.imread(data_path.format(case))
        result_image, deformation_field = registration(
                    fixed_image=atlas_image, moving_image=moving_image, 
                    method="affine", plot=False)  # ?Affine
        DVFs_list.append(deformation_field)


#%%
base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/"
T00_path = base_path+"train/image/case_0{}/T00.nii.gz"
T50_path = base_path+"train/image/case_0{}/T50.nii.gz"
cases = ["01", "02", "05", "06", "08"]
# cases = ["01", "02", "05", "06", "08", "22", "23", "24", "25"]  
 
# DVF = DVFs(atlas_image=A, data_path=T00_path, cases=cases)