
# %%
import itk
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import gryds
import scipy.ndimage as ndi

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

# use atlas image as fixed to see result better
img_atlas = nib.load(root_data+'atlas/atlasv1.nii.gz')
img_atlas = img_atlas.get_fdata()
img_atlas = itk.image_from_array(ndi.rotate((img_atlas).astype(np.float32),0)) # itk.itkImagePython.itkImageF3
plt.imshow(img_atlas[:,64,:], cmap="gray")

moving_image = img_data_T00[1]
fixed_image = img_data_T50[1]
fixed_image = img_atlas
#%% FROM DVF ELASTIX TO DVF GRYDS
# test scale necessary by comparing registration with transformation result
# 2 steps required: transpose and scaling

# Perform registration
parameter_path_base =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
parameter_file = "Par0007.txt"
result_image, DVF, result_transform_parameters = EF.registration(
    fixed_image=fixed_image, moving_image=moving_image,
    plot=True, parameter_path=parameter_path_base+parameter_file)

# Reshapre DVF from (160, 128, 160, 3) to (3, 160, 128, 160)
reshaped_DVF = np.transpose(np.asarray(DVF), (3, 0, 1, 2))  
if True: # TEST if np.transpose worked correctly
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(DVF[2,:, 64, :], cmap='gray')    
    axs[0].set_title("DVF[2]=CC")
    axs[1].imshow(DVF[0,64, :, :], cmap='gray')
    axs[1].set_title("DVF[0]=ML")
    axs[2].imshow(DVF[1,:, :, 64], cmap='gray')
    axs[2].set_title("DVF[1]=AP")
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(reshaped_DVF[:, 64, :,2], cmap='gray')    
    axs[0].set_title("DVF[2]=CC")
    axs[1].imshow(reshaped_DVF[64, :, :,0], cmap='gray')
    axs[1].set_title("DVF[0]=ML")
    axs[2].imshow(reshaped_DVF[:, :, 64,1], cmap='gray')
    axs[2].set_title("DVF[1]=AP")
    plt.tight_layout()
    plt.show()

# DVF elastix in pixels while DVF gryds in proportions. Therefore scale each direction with their pixelsize. 
# DVF[0] works medial-lateral on 160 pixels, DVF[2] cranial-caudial on 160 pixels, DVF[1] posterior-anterior in 128 pixels
DVF_scaled = np.asarray([reshaped_DVF[0]/160, reshaped_DVF[1]/128, reshaped_DVF[2]/160])

# gryds transformation takes DVFs in different order: CC, AP, ML
a_bspline_transformation = gryds.BSplineTransformation([DVF_scaled[2], DVF_scaled[1], DVF_scaled[0]])
an_image_interpolator = gryds.Interpolator(moving_image)
a_deformed_image = an_image_interpolator.transform(a_bspline_transformation)

if True: # TEST if scaling worked correctly
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(moving_image[:, 64, :], cmap='gray')
    axs[0, 0].set_title("moving image")
    axs[0, 1].imshow(DVF[:, 64, :, 2], cmap='gray')
    axs[0, 1].set_title("DVF")
    axs[1, 0].imshow(result_image[:, 64, :], cmap='gray')
    axs[1, 0].set_title("result form registration")
    axs[1, 1].imshow(a_deformed_image[:, 64, :], cmap='gray')
    axs[1, 1].set_title("result from interpolation")
    fig.suptitle("Comparing DVF from registration and translation - corodal")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(moving_image[64,:,:], cmap='gray')
    axs[0, 0].set_title("moving image")
    axs[0, 1].imshow(DVF[64,:,:, 0], cmap='gray')
    axs[0, 1].set_title("DVF")
    axs[1, 0].imshow(result_image[64,:,:], cmap='gray')
    axs[1, 0].set_title("result form registration")
    axs[1, 1].imshow(a_deformed_image[64,:, :], cmap='gray')
    axs[1, 1].set_title("result from interpolation")
    fig.suptitle("Comparing DVF from registration and translation - saggital")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(moving_image[:,:,64], cmap='gray')
    axs[0, 0].set_title("moving image")
    axs[0, 1].imshow(DVF[:,:,64, 1], cmap='gray')
    axs[0, 1].set_title("DVF")
    axs[1, 0].imshow(result_image[:,:,64], cmap='gray')
    axs[1, 0].set_title("result form registration")
    axs[1, 1].imshow(a_deformed_image[:,:,64], cmap='gray')
    axs[1, 1].set_title("result from interpolation")
    fig.suptitle("Comparing DVF from registration and translation - transverse")
    plt.tight_layout()
    plt.show()



# %% other attempts that did not work
#%% ATTEMPT OF APPLYING NUMPY DVF: itk functions 
# PROBLEM: IMAGE TYPES - was not able to fix
def deform_image(input_image, deformation_field, plot=False):
    print(type(input_image), type(deformation_field))
    # code adapted from https://examples.itk.org/src/filtering/imagegrid/warpanimageusingadeformationfield/documentation
    warpFilter = itk.WarpImageFilter[type(input_image), type(input_image), type(deformation_field)].New()
    # warpFilter = itk.WarpImageFilter[type(input_image), type(input_image), itk.Image[itk.Vector[itk.F, 3], 3]].New()
    # warpFilter = itk.WarpImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.Vector[itk.F, 3], 3]].New()

    interpolator = itk.LinearInterpolateImageFunction[type(input_image), itk.D].New()

    warpFilter.SetInterpolator(interpolator)

    warpFilter.SetOutputSpacing(input_image.GetSpacing())
    warpFilter.SetOutputOrigin(input_image.GetOrigin())
    warpFilter.SetOutputDirection(input_image.GetDirection())

    warpFilter.SetDisplacementField(deformation_field)

    warpFilter.SetInput(input_image)

    output_image = warpFilter.GetOutput()
    output_image.Update()
    if plot:
        EF.plot4(input_image, input_image, output_image, deformation_field)
    return output_image

#* Define image and DVF (average array and array)
input_image = itk.image_from_array(img_data_T00[0])
DVF_arr_avg = sum(np.asarray(DVFs)) / len(np.asarray(DVFs)) 
DVF_arr = np.asarray(DVFs[0])
# plt.imshow(DVFs_avg_array[:,64,:,2], cmap='gray')

#* ARRAY TO itkImageVF33 (itk.itkImagePython.itkImageVF33)       https://itkpythonpackage.readthedocs.io/en/master/Quick_start_guide.html 
DVF = itk.GetImageFromArray(np.ascontiguousarray(DVF_arr))   #itk.itkImagePython.itkImageF4
# DVF = itk.image_view_from_array(DVF_arr)                   #itk.itkImagePython.itkImageF4
# DVF = itk.image_from_array(DVF_arr, is_vector = True)      #itk.itkVectorImagePython.itkVectorImageF3
# DVF = itk.GetImageFromArray(DVF_arr)                       #itk.itkImagePython.itkImageF4
# DVF = itk.Image([itk.Vector[itk.F,3],3])
# DVF = itk.transformread(DVF_arr)                            # not supported
# DVF = itk.vnl_vector_from_array(DVF_arr)                    # not supported
# DVF = itk.matrix_from_array(DVF_arr)                        # not supported
# PixelType = itk.ctype("unsigned char")
# ImageType = itk.Image[itk.Vector[itk.F, 3], 3]
# np_view_array = itk.GetArrayViewFromImage(DVFs[0], ttype=ImageType)

if False:    
    # import argparse    
    # parser = argparse.ArgumentParser(description="Cast An Image To Another Type.")
    # parser.add_argument("input_image")
    # parser.add_argument("output_image")
    # args = parser.parse_args()

    DVF = itk.image_from_array(DVF_arr, is_vector = True)      #itk.itkVectorImagePython.itkVectorImageF3
    InputPixelType = itk.F
    InputImageType = itk.Image[itk.F, 4]

    OutputPixelType = itk.Vector[itk.F, 3]
    OutputImageType = itk.Image[itk.Vector[itk.F, 3], 3]

    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName("input_image")

    rescaler = itk.RescaleIntensityImageFilter[InputImageType, InputImageType].New()
    rescaler.SetInput(reader.GetOutput())
    rescaler.SetOutputMinimum(0)
    # outputPixelTypeMaximum = itk.NumericTraits[OutputPixelType].max()
    # rescaler.SetOutputMaximum(outputPixelTypeMaximum)

    castImageFilter = itk.CastImageFilter[InputImageType, OutputImageType].New()
    castImageFilter.SetInput(rescaler.GetOutput())

    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName("output_image")
    writer.SetInput(castImageFilter.GetOutput())

    writer.Update()

#* Excecute deformation
deform_image(input_image, DVFs[0], True)
deform_image(input_image, DVF, True)


#%% CHECK to see imported nii.gz has the same itk object type
path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/deformationField_case026.nii.gz"
img_DVF = nib.load(path)
img_DVF = img_DVF.get_fdata()
plt.imshow(img_DVF[:, 64, :,0,0], cmap='gray')
print(type(img_DVF), img_DVF.shape)
img_DVF = itk.image_from_array((img_DVF[:,:,:,0,:]).astype(np.float32))
print(type(img_DVF), img_DVF.shape)     # type(img_DVF)=itk.itkImagePython.itkImageF4
plt.imshow(img_DVF[0,:, 64, :], cmap='gray')
deform_image(input_image, img_DVF, True)

#%% ATTEMPT  OF APPLYING NUMPY DVF: opencv
# doesnt work either
# Load your 3D image and deformation field
# image_3d = np.zeros((160, 128, 160), dtype=np.uint8)  # Replace with your image data
# deformation_field = np.zeros((160, 128, 160, 3), dtype=np.float32)/100  # Replace with your deformation field data
image_3d = img_data_T00[0]
deformation_field = DVFs_avg_array*20
deformation_field = np.asarray(DVFs[0]).astype(np.float32)*10

# Create grid coordinates for the destination points
rows, cols, depths = image_3d.shape
x = np.arange(rows)
y = np.arange(cols)
z = np.arange(depths)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Add the deformation field to the destination points
dst_x = X + deformation_field[..., 0]
dst_y = Y + deformation_field[..., 1]
dst_z = Z + deformation_field[..., 2]

# Create the remap map
map_x = dst_x.astype(np.float32)
map_y = dst_y.astype(np.float32)
map_z = dst_z.astype(np.float32)

# Apply the remap operation to the 3D image
result = cv2.remap(image_3d[:,64,:], map_x[:,64,:], map_y[:,64,:], cv2.INTER_LINEAR)

# Display the result
plt.imshow(result, cmap='gray')

# plt.imshow(result[:,64,:], cmap='gray')
