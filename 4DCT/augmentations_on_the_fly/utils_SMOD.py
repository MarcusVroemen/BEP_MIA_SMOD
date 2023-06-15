import itk
import matplotlib.pyplot as plt
import nibabel as nib
import random
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import itk
import gryds
import nibabel as nib
import scipy.ndimage as ndi

plt.rcParams['image.cmap'] = 'gray'

# Prepair train data
def prepara_traindata(root_data):
    train_dataset = DatasetLung('train', root_data=root_data, augment_def=False, phases="in_ex")

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
    
    return img_data_T00, img_data_T50, img_data_T90

# Basic Elastix and Gryds functions
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
        # parameter_map = parameter_object.GetDefaultParameterMap(method)  # rigid
        # parameter_object.AddParameterMap(parameter_map)
        parameter_object.SetParameterMap(parameter_object.GetDefaultParameterMap(method))
    # print(parameter_object)

    # Registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object, #number_of_threads=8, 
        log_to_console=True, output_directory=output_directory)

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
    """Transform an image with a DVF using the gryds library
    Since the gryds format for DVFs is different to ITK, first the DVF is scaled
    Args:
        DVF_itk: DVF as array or itk object with displacements in pixels
            (converted to displacments scaled to image size with DVF_conversion)
        img_moving: to transform image
    """
    DVF_gryds = DVF_conversion(DVF_itk)

    bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
    an_image_interpolator = gryds.Interpolator(img_moving)
    img_deformed = an_image_interpolator.transform(bspline_transformation)
    if plot:
        plot_registration(fixed_image=img_moving, moving_image=img_deformed, deformation_field=DVF_itk, full=True, 
                            result_image=np.asarray(img_moving) - np.asarray(img_deformed), title="transformation with averaged atlas DVF",
                            name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")
    return img_deformed

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
        fig.suptitle("In- and output of registration - frontal")

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

def validate_DVFs(DVFs, DVFs_inverse, img_moving):
    """Plot DVFs and inverse with effect on images to validate them
    Args:
        DVFs: itk or array type DVFs that register imgs to atlas
        DVFs_inverse: inverse itk or array type DVFs that
        img_moving: an image where the DVFs are applied to
    """
    fig, ax = plt.subplots(len(DVFs),5, figsize=(40,50))
    for i in range(len(DVFs)):
        # DVF and img+DVF
        ax[i,0].imshow(np.asarray(DVFs[i])[:,64,:,2])
        img_ogDVF = transform(DVF_itk=DVFs[i], img_moving=img_moving, plot=False)
        ax[i,1].imshow(img_ogDVF[:,64,:])
        ax[i,1].set_title("original deformed with DVF")
        # DVFinverse and img+DVFinverse
        ax[i,2].imshow(np.asarray(DVFs_inverse[i])[:,64,:,2])
        img_invDVF = transform(DVF_itk=DVFs_inverse[i], img_moving=img_moving, plot=False)
        ax[i,3].imshow(img_invDVF[:,64,:])
        ax[i,3].set_title("original deformed with DVFinv")
        # img+DVFinverse+DVF
        img_3 = transform(DVF_itk=DVFs[i], img_moving=img_invDVF, plot=False)
        ax[i,4].imshow(img_3[:,64,:])
        ax[i,4].set_title("original deformed with DVFinf and DVF")
    plt.tight_layout()
    plt.show()

# Atlas generation
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
    
    print("Atlas generation complete/n")
    return An

def register_to_atlas(img_data, img_atlas, root_data, method="affine", plot=True, inverse=True):
    """Generate DVFs from set images registered on atlas image"""
    params_path = root_data.replace("data","transform_parameters")
    DVFs_list, DVFs_inverse_list, imgs_to_atlas = [], [], []

    for i in range(len(img_data)):
        result_image, DVF, result_transform_parameters = registration(
            fixed_image=img_atlas, moving_image=img_data[i], 
            method=method, plot=plot, 
            output_directory=params_path)
        DVFs_list.append(DVF)
        imgs_to_atlas.append(result_image)
        
        # Inverse DVF
        if inverse:
            parameter_object = itk.ParameterObject.New()
            parameter_map = parameter_object.GetDefaultParameterMap(method)
            parameter_map['HowToCombineTransforms'] = ['Compose']
            parameter_object.AddParameterMap(parameter_map)
            inverse_image, inverse_transform_parameters = itk.elastix_registration_method(
                img_data[i], img_data[i],
                parameter_object=parameter_object,
                initial_transform_parameter_file_name=params_path+"TransformParameters.0.txt")
            inverse_transform_parameters.SetParameter(
                0, "InitialTransformParametersFileName", "NoInitialTransform")
            DVF_inverse = itk.transformix_deformation_field(img_data[i], inverse_transform_parameters)
            DVFs_inverse_list.append(DVF_inverse)
        
        
    return DVFs_list, DVFs_inverse_list, imgs_to_atlas

# DVF generation
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

def generate_artificial_DVFs(DVFs_artificial_components, num_images, sigma):
    """Generate artificial DVFs from list with to_atlas_DVFs
    Args:
        DVFs_artificial_components (list with arrays): DVF_mean, DVF_Ud from dimreduction()
        num_images (int): amount of artificial DVFs to generate
        sigma (int): random scaling component 100: visual deformations, 500: too much
        DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
    Returns:
        DVFs_artificial (list with arrays): artificial DVFs with shape (160, 128, 160, 3)
    """
    # Unpack mean and PCA components from dimreduction()
    DVF_mean, DVF_Ud = DVFs_artificial_components
    
    DVFs_artificial = []
    for i in range(num_images):
        DVF_artificial = np.zeros((DVF_mean[0].shape[0], 3))
        for j in range(3):
            x = np.random.normal(loc=0, scale=sigma, size=DVF_Ud[j].shape[0])
            # Paper: vg = Vmean + U*x*d 
            DVF_artificial[:,j] = DVF_mean[j] + np.dot(DVF_Ud[j].T, x)  #(3276800,1)
        DVFs_artificial.append(np.reshape(DVF_artificial, (160, 128, 160, 3)))

    return DVFs_artificial

# IMG generation
def generate_artificial_imgs(imgs_to_atlas, DVFs_artificial_inverse, img_data="", plot=False, breathing=False):
    """Use DVFs and images registered to atlas to generate artificial images
    Args:
        imgs_to_atlas: images generated by registering them to atlas image
        DVFs_artificial_inverse: itk or array type DVFs from generate_artificial_DVFs() function
        img_data: original images, only used for visualization
        plot: plots original, to-atlas and artificial image
    """
    imgs_artificial = []
    counter=0
    j=0
    for i in range(len(imgs_to_atlas)):
        if breathing==False:
            for j in range(len(DVFs_artificial_inverse)):
                img_artificial = transform(DVF_itk=DVFs_artificial_inverse[j], img_moving=imgs_to_atlas[i], plot=False)
                imgs_artificial.append(img_artificial)
                counter+=1
                print("Image {}/{} created".format(counter, len(imgs_to_atlas)*len(DVFs_artificial_inverse)))
                if plot:
                    plot_registration(fixed_image=img_data[i], moving_image=imgs_to_atlas[i], result_image=img_artificial, deformation_field=DVFs_artificial_inverse[j],
                                        name1="T00 original", name2="T00 to atlas", name3="artificial T00", name4="artificial DVF",
                                        title="Creating artificial images", full=True)
        else:
            # When generating artificial T50a, we want 1 T50a for every T10a where there are n times more artificial DVFs
            # Therefore dont also loop over DVFs
            img_artificial = transform(DVF_itk=DVFs_artificial_inverse[j], img_moving=imgs_to_atlas[i], plot=False)
            j+=1
            if j==len(DVFs_artificial_inverse):
                j=0
            imgs_artificial.append(img_artificial)
            
            counter+=1
            
            print("Image {}/{} created".format(counter, len(imgs_to_atlas)))
            if plot:
                EF.plot_registration(fixed_image=img_data[i], moving_image=imgs_to_atlas[i], result_image=img_artificial, deformation_field=DVFs_artificial_inverse[j],
                                    name1="T00 original", name2="T00 to atlas", name3="artificial T00", name4="artificial DVF",
                                    title="Creating artificial images", full=True)
    return imgs_artificial 

# MAIN functions  
def data_augm_preprocessing(img_data, root_data, generate=True):
    """Preprocessing of data augmentation which created the components for artificial DVFs and imgs_to_atlas
    Split from the data_augm_generation since these components only have to be made once which can be done as preprocessing step
    Function includes: (1) atlas generation, (2) registration to atlas, (3) calculate components for artificial DVFs
    Args:
        generate (bool): Whether to generate atlas or load it in #*temporaty
        img_data (list with itk images): original training data to augment
    Returns:
        DVFs_artificial_components: components necessary for DVF generation (DVF_mean, DVF_Ud), does not contain random component yet
        imgs_to_atlas (list with itk images): original training data registered to atlas
    """
    # (1) Generate or get atlas image
    print("Generating atlas")
    if generate:
        # Generate atlas image
        img_atlas = generate_atlas(img_data=img_data)
        plt.imshow(img_atlas[:,64,:], cmap="gray")
        # Save atlas image
        img = nib.load(root_data+'train/image/case_001/T00.nii.gz')
        img_atlas_nib = nib.Nifti1Image(img_atlas, img.affine)
        nib.save(img_atlas_nib, os.path.join(root_data,'atlas', 'atlasv1.nii.gz'))
    else:
        # Load already generated atlas image
        img_atlas = nib.load(root_data+'atlas/atlasv1.nii.gz')
        img_atlas = img_atlas.get_fdata()
        img_atlas = itk.image_from_array(ndi.rotate((img_atlas).astype(np.float32),0)) # itk.itkImagePython.itkImageF3
        plt.imshow(img_atlas[:,64,:], cmap="gray")
    
    # (2) Register to atlas for DVFinverse and imgs_to_atlas  
    print("Generating artificial images")  
    DVFs, DVFs_inverse, imgs_to_atlas = register_to_atlas(img_data=img_data, img_atlas=img_atlas)
    
    # Optionally validate DVFs and DVFs_inverse (uncomment next line)
    # validate_DVFs(DVFs, DVFs_inverse, img_moving=img_data[0])
    
    # (3) Get components needed for artificial DVF generation
    print("Generating artificial DVF components")
    DVFs_artificial_components = dimreduction(DVFs=DVFs_inverse)
    
    return DVFs_artificial_components, imgs_to_atlas, DVFs

def data_augm_generation(DVFs_artificial_components, imgs_to_atlas=None, img_data=None, sigma=1500, num_images=3, DVFs=None, imgs_T50=None, plot=True):
    """Generate artificial training data with on the spot
    Contains random component sigma so imgs_artificial are never the same
    Args:
        DVFs_artificial_components: components necessary for DVF generation (DVF_mean, DVF_Ud) from dimreduction()
        imgs_to_atlas (list with itk images): original training data registered to atlas
        img_data: original training data (only needed) when plotting the generated images
        sigma: random component in artificial DVF generation (500 gives noticable differences)
        
    """
    # Generate artificial DVFs
    print("Generating artificial DVFs")
    DVFs_artificial = generate_artificial_DVFs(DVFs_artificial_components=DVFs_artificial_components, 
                                               num_images=num_images, sigma=sigma)

    if plot:
        img_grid = np.zeros((160, 128, 160))
        line_interval = 5  # Adjust this value to change the interval between lines
        img_grid[:, ::line_interval, :] = 1
        img_grid[:, :, ::line_interval] = 1
        img_grid[::line_interval, :, :] = 1
        for i in range(len(DVFs_artificial)):
            transform(DVFs_artificial[i], img_grid, plot=True)
    
    # generate artificial images
    print("Generating artificial images")
    # if imgs_to_atlas!=None:
    imgs_artificial_T00 = generate_artificial_imgs(imgs_to_atlas=imgs_to_atlas, DVFs_artificial_inverse=DVFs_artificial, img_data=img_data, plot=False)
    
    if imgs_T50!=None:
        print("Transform T50 images with same dvfs")
        # Register T50 images to atlas with DVFs_T00
        imgs_to_atlas_T50 = []
        for i in range(len(img_data)):
            img_to_atlas_T50 = transform(DVF_itk=DVFs[i], img_moving=img_data[i], plot=False)
            imgs_to_atlas_T50.append(img_to_atlas_T50)  
        imgs_artificial_T50 = generate_artificial_imgs(imgs_to_atlas=imgs_to_atlas_T50, DVFs_artificial_inverse=DVFs_artificial, img_data=img_data, plot=False)
    
        return imgs_artificial_T00, imgs_artificial_T50
    else: 
        return imgs_artificial_T00

def data_augm_breathing(imgs_T00, imgs_T50, imgs_T00a, sigma=1000, num_images=3, plot=False):
    # register T00 to T50 with bspline to get the breathing motion
    DVFs_breathing=[]
    for i in range(len(imgs_T00)):
        result_image, DVF_breathing, result_transform_parameters = registration(
            fixed_image=imgs_T50[i], moving_image=imgs_T00[i], 
            method="bspline", plot=True)#, parameter_path=parameter_path_base+parameter_file)
        DVFs_breathing.append(DVF_breathing)

    # generate artificial DVFs that model breathing motion from T00 to T50
    print("Generating artificial DVFs - breathing motion")
    DVFs_artificial_components_breathing = dimreduction(DVFs=DVFs_breathing)
    DVFs_artificial_breathing = generate_artificial_DVFs(DVFs_artificial_components=DVFs_artificial_components_breathing, 
                                            num_images=num_images, sigma=sigma)
    
    if plot: #plot dvf on grid
        img_grid = np.zeros((160, 128, 160))
        line_interval = 5  # Adjust this value to change the interval between lines
        img_grid[:, ::line_interval, :] = 1
        img_grid[:, :, ::line_interval] = 1
        img_grid[::line_interval, :, :] = 1
        for i in range(len(DVFs_artificial_breathing)):
            transform(DVFs_artificial_breathing[i], img_grid, plot=True)

    # apply artificial breathing motion to T00a
    imgs_artificial_T50 = generate_artificial_imgs(imgs_to_atlas=imgs_T00a, DVFs_artificial_inverse=DVFs_artificial_breathing, plot=False, breathing=True)
    return imgs_artificial_T50

# Plotting and writing final data
def plot_data_augm(imgs_artificial, imgs_original, num_images, title, neg=False):
    num_rows = (len(imgs_artificial) + 2) // num_images        
    fig, axes = plt.subplots(num_rows, num_images+1, figsize=(num_images*3, num_rows*2.5))        
    fig.suptitle(title, y=1)  
    i_original=0
    i_artificial=0  
    for i, ax in enumerate(axes.flatten()):
        if i%(num_images+1)==0:
            ax.imshow(imgs_original[i_original][:,64,:])
            ax.axis('off')
            ax.set_title("Original image", fontsize=15)
            i_original+=1
        elif neg==False:
            ax.imshow(imgs_artificial[i_artificial][:,64,:])
            ax.axis('off')
            ax.set_title("Artificial image", fontsize=15)
            i_artificial+=1  
        elif neg==True:
            imgs=(imgs_artificial[i_artificial]) - np.asarray(imgs_original[i_original-1])
            ax.imshow(imgs[:,64,:])
            ax.axis('off')
            ax.set_title("Artificial - original", fontsize=13)
            i_artificial+=1    
    plt.tight_layout()    
    plt.plot()

def plot_data_augmpairs(imgs_artificial_1, imgs_artificial_2, title):
    fig, axes = plt.subplots(len(imgs_artificial_1), 3, figsize=(3*2.5, len(imgs_artificial_1)*3))        
    fig.suptitle(title, y=1)  
    for i in range(len(imgs_artificial_1)):
        axes[i,0].imshow(imgs_artificial_1[i][:,64,:])
        axes[i,1].imshow(imgs_artificial_2[i][:,64,:])
        img_neg=np.asarray(imgs_artificial_2[i]) - np.asarray(imgs_artificial_1[i])
        axes[i,2].imshow(img_neg[:,64,:])
        axes[i,0].axis('off')
        axes[i,1].axis('off')
        axes[i,2].axis('off') 
    plt.tight_layout()    
    plt.plot()

def write_augmented_data(path, foldername, imgs_T00a, imgs_T50a):
    img = nib.load(path+'train/image/case_001/T00.nii.gz')
    for i in range(len(imgs_T00a)):
        folder_path = os.path.join(path,foldername,"image", str(i).zfill(3))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        img_nib = nib.Nifti1Image(imgs_T00a[i], img.affine)
        nib.save(img_nib, os.path.join(folder_path, 'T00.nii.gz'))
        img_nib = nib.Nifti1Image(imgs_T50a[i], img.affine)
        nib.save(img_nib, os.path.join(folder_path, 'T50.nii.gz'))
        

if __name__ == "__main__":
    # root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
    # root_data = 'C:/Users/Quinten Vroemen/Documents/MV_codespace/BEP_MIA_DIR/4DCT/data/'
    root_data = '/home/bme001/20203531/BEP/BEP_MIA_DIR/BEP_MIA_DIR/4DCT/data/'
    augmentation=True
    # Number of images to generate per training image
    NUM_IMAGES_TO_GENERATE = 3

    # Random component
    factor=15
    SIGMA1=1000*factor
    SIGMA2=100*factor

    if augmentation:
        img_data_T00, img_data_T50, img_data_T90 = prepara_traindata(root_data=root_data)

        # STEP 1
        DVFs_artificial_components_T00, imgs_to_atlas_T00, DVFs_T00 = data_augm_preprocessing(generate=False, img_data=img_data_T00) # saved atlas is T00
        imgs_artificial_T00 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, imgs_to_atlas=imgs_to_atlas_T00, 
                        img_data=img_data_T00, sigma=SIGMA1, num_images=NUM_IMAGES_TO_GENERATE, plot=False)
        # plot_data_augm(imgs_artificial=imgs_artificial_T00, imgs_original=img_data_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00s (DVFaT00) sigma={}".format(SIGMA1), neg=True)
        # plot_data_augm(imgs_artificial=imgs_artificial_T00, imgs_original=img_data_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00s (DVFaT00) sigma={}".format(SIGMA1), neg=False)
        
        # STEP 2
        imgs_artificial_T50 = data_augm_breathing(imgs_T00=img_data_T00, imgs_T50=img_data_T50, imgs_T00a=imgs_artificial_T00, 
                                                    sigma=SIGMA2, num_images=NUM_IMAGES_TO_GENERATE, plot=False)
        # plot_data_augm(imgs_artificial=imgs_artificial_T50, imgs_original=imgs_artificial_T50, num_images=NUM_IMAGES_TO_GENERATE, title="T50aEXP (DVFa-EXP with bspline on T00a) sigma={}".format(SIGMA2), neg=False)
        # plot_data_augm(imgs_artificial=imgs_artificial_T50, imgs_original=imgs_artificial_T50, num_images=NUM_IMAGES_TO_GENERATE, title="T50aEXP (DVFa-EXP with bspline on T00a) sigma={}".format(SIGMA2), neg=True)
        
        plot_data_augmpairs(imgs_artificial_T00, imgs_artificial_T50, title="T00a and T50a with same DVFa sigma={}|{}".format(SIGMA1, SIGMA2))
        # plot_data_augmpairs(img_data_T00, img_data_T50, title="T00 and T50 original")

        # Write data
        write_augmented_data(path=root_data, foldername="artificial/artificial_N{}_S{}_{}".format(NUM_IMAGES_TO_GENERATE, SIGMA1, SIGMA2), 
                                imgs_T00a=imgs_artificial_T00, imgs_T50a=imgs_artificial_T50)

    else:
        n=10
        factor=20
        s1=factor*1000
        s2=factor*100
        folder_augment=f"artificial_N{n}_S{s1}_{s2}"
        
        train_dataset = DU.DatasetLung('artificial', root_data=root_data, folder_augment=folder_augment, augment_def=False, phases="in_ex")
        img_data_T00a, img_data_T50a = [], []
        for i in range(len(train_dataset)):
            img_fixed, img_moving,_,_ = train_dataset[i]
            img_data_T00a.append(ndi.rotate(img_fixed.squeeze(0).numpy(),-90,axes=(2,0))), img_data_T50a.append(ndi.rotate(img_moving.squeeze(0).numpy(),-90,axes=(2,0)))  #fixed T50 are dubble, moving T00a and T90       

        plot_data_augmpairs(img_data_T00a, img_data_T50a, title="T00a and T50a N={} sigma={}|{}".format(n, s1, s1))
        

