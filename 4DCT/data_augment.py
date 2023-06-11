import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
import itk
import gryds
import nibabel as nib
import scipy.ndimage as ndi

import datasets_utils as DU 
import elastix_functions as EF
plt.rcParams['image.cmap'] = 'gray'


#* functions #########################
def prepara_traindata(root_data):
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
    
    return img_data_T00, img_data_T50, img_data_T90

def register_to_atlas(img_data, img_atlas, method="affine", plot=True, inverse=True):
    """Generate DVFs from set images registered on atlas image"""
    params_path="transform_parameters/" #!
    DVFs_list, DVFs_inverse_list, imgs_to_atlas = [], [], []

    for i in range(len(img_data)):
        #!
        # parameter_path_base =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
        # parameter_file = "Par0007.txt"
    
        result_image, DVF, result_transform_parameters = EF.registration(
            fixed_image=img_atlas, moving_image=img_data[i], 
            method=method, plot=plot, 
            output_directory=params_path) #, parameter_path=parameter_path_base+parameter_file
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

def generate_artificial_imgs(imgs_to_atlas, DVFs_artificial_inverse, img_data="", plot=False):
    """Use DVFs and images registered to atlas to generate artificial images
    Args:
        imgs_to_atlas: images generated by registering them to atlas image
        DVFs_artificial_inverse: itk or array type DVFs from generate_artificial_DVFs() function
        img_data: original images, only used for visualization
        plot: plots original, to-atlas and artificial image
    """
    imgs_artificial = []
    for i in range(len(imgs_to_atlas)):
        for j in range(len(DVFs_artificial_inverse)):
            img_artificial = transform(DVF_itk=DVFs_artificial_inverse[j], img_moving=imgs_to_atlas[i], plot=False)
            imgs_artificial.append(img_artificial)
            if plot:
                EF.plot_registration(fixed_image=img_data[i], moving_image=imgs_to_atlas[i], result_image=img_artificial, deformation_field=DVFs_artificial_inverse[j],
                                    name1="T00 original", name2="T00 to atlas", name3="artificial T00", name4="artificial DVF",
                                    title="Creating artificial images", full=True)
    return imgs_artificial   

# Large functions
def data_augm_preprocessing(img_data, generate=True):
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
        img_atlas = EF.generate_atlas(img_data=img_data)
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

def data_augm_generation(DVFs_artificial_components, imgs_to_atlas=None, img_data=None, sigma=500, num_images=3, DVFs=None):
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

    # generate artificial images
    print("Generating artificial images")
    if imgs_to_atlas!=None:
        imgs_artificial = generate_artificial_imgs(imgs_to_atlas=imgs_to_atlas, DVFs_artificial_inverse=DVFs_artificial, img_data=img_data, plot=False)
    else:
        # imgs_to_atlas is not define the goal is to apply DVF_artificial_T00 to T50 images
        imgs_to_atlas_alternative = []
        for i in range(len(img_data)):
            img_to_atlas_alternative = transform(DVF_itk=DVFs[i], img_moving=img_data[i], plot=False)
            imgs_to_atlas_alternative.append(img_to_atlas_alternative)    
        imgs_artificial = generate_artificial_imgs(imgs_to_atlas=img_to_atlas_alternative, DVFs_artificial_inverse=DVFs_artificial, img_data=img_data, plot=False)
        
    return imgs_artificial

def plot_data_augm(imgs_artificial, num_images, title, sigma):
    num_rows = (len(imgs_artificial) + 2) // num_images        
    fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images*3, num_rows*3))        
    fig.suptitle("Augmented {} images with sigma={}".format(title, sigma), y=1)    
    for i, ax in enumerate(axes.flatten()):
        if i < len(imgs_artificial):
            ax.imshow(imgs_artificial[i][:,64,:])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()    
    plt.plot()

if __name__ == "__main__":
    root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
    root_data = 'C:/Users/Quinten Vroemen/Documents/MV_codespace/BEP_MIA_DIR/4DCT/data/'
    img_data_T00, img_data_T50, img_data_T90 = prepara_traindata(root_data=root_data)

    # Data to perform augmentation on
    IMG_DATA = img_data_T00
    # Number of images to generate per training image
    NUM_IMAGES_TO_GENERATE = 3
    # Random component
    SIGMA=15000

    mode="test"
    # Data augmentation
    if mode=="a":
        """ARTIFICIAL PATIENTS 1: Get DVFaT00 from T00->atlasT00->PCA apply to T00 | DVFaT50 from T50->atlasT50->PCA apply to T50"""
        DVFs_artificial_components_T00, imgs_to_atlas_T00, _ = data_augm_preprocessing(generate=False, img_data=img_data_T00) # saved atlas is T00
        imgs_artificial_T00 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, imgs_to_atlas=imgs_to_atlas_T00, 
                        img_data=img_data_T00, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)

        DVFs_artificial_components_T50, imgs_to_atlas_T50, _ = data_augm_preprocessing(generate=True, img_data=img_data_T50)
        imgs_artificial_T50 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T50, imgs_to_atlas=imgs_to_atlas_T50, 
                        img_data=img_data_T50, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)
        
        plot_data_augm(imgs_artificial=imgs_artificial_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00)", sigma=SIGMA)
        plot_data_augm(imgs_artificial=imgs_artificial_T50, num_images=NUM_IMAGES_TO_GENERATE, title="T50 (DVFaT50)", sigma=SIGMA)
    
        imgs_artificial_DIF=[]
        for i in range(len(imgs_artificial_T00)):
            imgs_artificial_DIF.append(np.asarray(imgs_artificial_T00[i])-np.asarray(imgs_artificial_T50[i]))
        plot_data_augm(imgs_artificial=imgs_artificial_DIF, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00) - T50 (DVFaT50)", sigma=SIGMA)


    elif mode=="b":
        """ARTIFICIAL PATIENTS 2: Get DVFaT00 from T00->atlas->PCA and apply it to T00 and T50 for new image pairs"""
        DVFs_artificial_components_T00, imgs_to_atlas_T00, DVFs_T00 = data_augm_preprocessing(generate=False, img_data=img_data_T00)
        imgs_artificial_T00 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, imgs_to_atlas=imgs_to_atlas_T00, 
                        img_data=img_data_T00, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)
        
        imgs_artificial_T50 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, 
                        img_data=img_data_T50, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE, DVFs=DVFs_T00)
        
        plot_data_augm(imgs_artificial=imgs_artificial_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00)", sigma=SIGMA)
        plot_data_augm(imgs_artificial=imgs_artificial_T50, num_images=NUM_IMAGES_TO_GENERATE, title="T50 (DVFaT00)", sigma=SIGMA)

        imgs_artificial_DIF=[]
        for i in range(len(imgs_artificial_T00)):
            imgs_artificial_DIF.append(np.asarray(imgs_artificial_T00[i])-np.asarray(imgs_artificial_T50[i]))
        plot_data_augm(imgs_artificial=imgs_artificial_DIF, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00) - T50 (DVFaT00)", sigma=SIGMA)


    elif mode=="c":
        """ARTIFICIAL EXPIRATION: Get DVFaT00 from T00->atlas->PCA and DVFaT50 from T50->atlas->PCA and both apply to T00"""
        DVFs_artificial_components_T00, imgs_to_atlas_T00, _ = data_augm_preprocessing(generate=False, img_data=img_data_T00) # saved atlas is T00
        imgs_artificial_T00 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, imgs_to_atlas=imgs_to_atlas_T00, 
                        img_data=img_data_T00, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)

        DVFs_artificial_components_T50, imgs_to_atlas_T50, _ = data_augm_preprocessing(generate=True, img_data=img_data_T50)
        
        #apply DVFaT50 to T00a for artificial expiration
        #3*3 synthetic T50a are created for every 3 T00a
        imgs_artificial_T00_EXP = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T50, imgs_to_atlas=imgs_artificial_T00, 
                        img_data=imgs_artificial_T00, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)
        
        plot_data_augm(imgs_artificial=imgs_artificial_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00)", sigma=SIGMA)
        plot_data_augm(imgs_artificial=imgs_artificial_T00_EXP, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00+DVFaT50)", sigma=SIGMA)
        
        imgs_artificial_DIF=[]
        for i in range(len(imgs_artificial_T00_EXP)):
            imgs_artificial_DIF.append(np.asarray(imgs_artificial_T00[i//NUM_IMAGES_TO_GENERATE])-np.asarray(imgs_artificial_T00_EXP[i]))
        plot_data_augm(imgs_artificial=imgs_artificial_DIF, num_images=NUM_IMAGES_TO_GENERATE, title="T00 (DVFaT00) - T00 (DVFaT00+DVFaT50)", sigma=SIGMA)

    elif mode=="test":
        DVFs_artificial_components_T00, imgs_to_atlas_T00, _ = data_augm_preprocessing(generate=False, img_data=img_data_T00) # saved atlas is T00
        imgs_artificial_T00 = data_augm_generation(DVFs_artificial_components=DVFs_artificial_components_T00, imgs_to_atlas=imgs_to_atlas_T00, 
                        img_data=img_data_T00, sigma=SIGMA, num_images=NUM_IMAGES_TO_GENERATE)
        plot_data_augm(imgs_artificial=imgs_artificial_T00, num_images=NUM_IMAGES_TO_GENERATE, title="T00s (DVFaT00)", sigma=SIGMA)
        
        
        #register T00 to T50 with bspline to get the breathing motion
        DVFs_EXP=[]
        for i in range(len(img_data_T00)):
            parameter_path_base =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
            parameter_file = "Par0007.txt"
            result_image, DVF_EXP, result_transform_parameters = EF.registration(
                fixed_image=img_data_T50[i], moving_image=img_data_T00[i], 
                method="affine", plot=True, parameter_path=parameter_path_base+parameter_file)
            DVFs_EXP.append(DVF_EXP)

        # generate artificial DVFs that model breathing motion from T00 to T50
        DVFs_artificial_components_EXP = dimreduction(DVFs=DVFs_EXP)
        DVFs_artificial_EXP = generate_artificial_DVFs(DVFs_artificial_components=DVFs_artificial_components_EXP, 
                                               num_images=NUM_IMAGES_TO_GENERATE, sigma=SIGMA)

        # apply breathing motion to artificial T00s
        imgs_artificial = generate_artificial_imgs(imgs_to_atlas=imgs_artificial_T00, DVFs_artificial_inverse=DVFs_artificial_EXP, img_data=img_data_T00, plot=True)

        plot_data_augm(imgs_artificial=imgs_artificial, num_images=NUM_IMAGES_TO_GENERATE, title="T50a (DVFa-insp on T00a)", sigma=10000)
