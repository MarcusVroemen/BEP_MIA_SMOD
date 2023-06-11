import itk
import matplotlib.pyplot as plt
import nibabel as nib
import random
from matplotlib import pyplot as plt
import numpy as np

def registration(fixed_image, moving_image, method="rigid", plot=False, 
                 parameter_path=None, output_directory=""):
    """Function that calls Elastix registration function
    Args:
        fixed_image: list with arrays of images (same as moving_image)
        method: string with either one of the standard registration methods
               or the name of a parameter file
    """
        
    # print("Registration step start ")

    # Define parameter object
    parameter_object = itk.ParameterObject.New()
    if parameter_path != None:
        parameter_object.AddParameterFile(parameter_path)    #e.g. Par007.txt
    else:
        parameter_map = parameter_object.GetDefaultParameterMap(method)  # rigid
        parameter_object.AddParameterMap(parameter_map)
    # print(parameter_object)

    # Registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        number_of_threads=8, log_to_console=True, output_directory=output_directory)

    # Deformation field
    deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
    print("Registration step complete")

    # Jacobanian #? does not work always
    # jacobians = itk.transformix_jacobian(moving_image, result_transform_parameters)
    # # Casting tuple to two numpy matrices for further calculations.
    # spatial_jacobian = np.asarray(jacobians[0]).astype(np.float32)
    # det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float32)
    # print("Number of foldings in transformation:",np.sum(det_spatial_jacobian < 0))
    
    if plot:
        plot_registration(fixed_image, moving_image, result_image, deformation_field, full=True)

    return result_image, deformation_field, result_transform_parameters

def plot_registration(fixed_image, moving_image, result_image, deformation_field,
          name1="fixed image", name2="moving image", name3="result image", name4="deformation field",
          title="In- and output of registration - frontal/transverse/saggital" , full=True):
    """Plot fixed, moving result image and deformation field
       Called after registration to see result"""
    
    if full==False:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(fixed_image[:, 64, :], cmap='gray')
        axs[0, 0].set_title(name1)
        axs[0, 1].imshow(moving_image[:, 64, :], cmap='gray')
        axs[0, 1].set_title(name2)
        axs[1, 0].imshow(result_image[:, 64, :], cmap='gray')
        axs[1, 0].set_title(name3)
        axs[1, 1].imshow(deformation_field[:, 64, :, 2], cmap='gray')
        axs[1, 1].set_title(name4)

    elif full==True:
        fig, axs = plt.subplots(2, 6, figsize=(30, 10), dpi=80) #, dpi=80
        plt.rc('font', size=20) 
        axs[0, 0].imshow(fixed_image[:, 64, :], cmap='gray')
        axs[0, 0].set_title(name1)
        axs[0, 1].imshow(moving_image[:, 64, :], cmap='gray')
        axs[0, 1].set_title(name2)
        axs[1, 0].imshow(result_image[:, 64, :], cmap='gray')
        axs[1, 0].set_title(name3)
        axs[1, 1].imshow(deformation_field[:, 64, :, 2], cmap='gray')
        axs[1, 1].set_title(name4)
        
        axs[0, 2].imshow(fixed_image[64, :, :], cmap='gray')
        axs[0, 2].set_title(name1)
        axs[0, 3].imshow(moving_image[64, :, :], cmap='gray')
        axs[0, 3].set_title(name2)
        axs[1, 2].imshow(result_image[64, :, :], cmap='gray')
        axs[1, 2].set_title(name3)
        axs[1, 3].imshow(deformation_field[64, :, :, 0], cmap='gray')
        axs[1, 3].set_title(name4)
        
        axs[0, 4].imshow(fixed_image[:, :, 64], cmap='gray')
        axs[0, 4].set_title(name1)
        axs[0, 5].imshow(moving_image[:, :, 64], cmap='gray')
        axs[0, 5].set_title(name2)
        axs[1, 4].imshow(result_image[:, :, 64], cmap='gray')
        axs[1, 4].set_title(name3)
        axs[1, 5].imshow(deformation_field[:, :, 64, 1], cmap='gray')
        axs[1, 5].set_title(name4)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def generate_atlas(img_data):
    """Generate atlas image from list of images

    Args:
        img_data: list with arrays of images

    Returns:
        An: Atlas image as itk image type
    """
    # Initialize atlas A0 - Select random train image
    random_image = random.randint(0, len(img_data)-1)
    print("1. Initializing atlas A0 as image {}".format(random_image))
    A0 = img_data[random_image]
    
    # Rigidly register each image to A0 to obtain registered set IT0
    print("2. Registration of initial image set IT0 to A0 to acquire A1")
    registered_set = []  # IT0
    for img in img_data:
        moving_image = img
        
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
        A_new = itk.image_from_array(A_new_array.astype(np.float32))
        
        plot_registration(An, A_new, np.asarray(An) - np.asarray(A_new), deformation_field, full=False)
        
        iteration += 1

        # Calculate L2 norm and break when convergence is reached
        L2 = np.linalg.norm(np.asarray(An) - np.asarray(A_new)) 
        print("L2: {}   dL2: {}".format(L2, L2_last - L2))
        
        if (L2_last - L2) <= 1:
            break
    
    print("Atlas generation complete")
    return An