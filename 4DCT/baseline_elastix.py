# %%
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import random
import torch
import itk
import nibabel as nib
# import SimpleITK as sitk
# import elastix
# import os
# import imageio
# import scipy.stats as stats
# import openpyxl

import datasets_utils as DU 
import elastix_functions as EF
from itkwidgets import view


#%% Load data
def get_img_data(train_val_test="train", root_data='C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'):
    dataset = DU.DatasetLung(train_val_test, root_data=root_data, augment_def=False, phases="in_ex")

    if train_val_test == "train":
        img_data_T00, img_data_T50, img_data_T90 = [], [], []
        for i in range(len(dataset)):
            img_fixed, img_moving,_,_ = dataset[i]
            # image pairs returned [00,50] and [90,50] returned, therefore new image can be assigned half of the time
            if i%2==0:
                img_data_T00.append(img_fixed.squeeze(0).numpy()), img_data_T50.append(img_moving.squeeze(0).numpy())  #fixed T50 are dubble, moving T00 and T90       
            else:           
                img_data_T90.append(img_fixed.squeeze(0).numpy())

        return img_data_T00, img_data_T50, img_data_T90

    elif train_val_test != "train":
        img_data_T00, img_data_T50 = [], []
        pts_data_T00, pts_data_T50 = [], []
        for i in range(len(dataset)):
            img_fixed, img_moving,_,_ = dataset[i]
            pts_moving, pts_fixed = dataset.get_landmarks(i)
            # Only image pairs [00,50] pairs are returned so can be assigned directly
            img_data_T00.append(img_fixed.squeeze(0).numpy()), img_data_T50.append(img_moving.squeeze(0).numpy())
            pts_data_T00.append(pts_fixed.numpy()), pts_data_T50.append(pts_moving.numpy())

        return img_data_T00, img_data_T50, pts_data_T00, pts_data_T50

root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'

img_traindata_T00, img_traindata_T50, img_traindata_T90 = get_img_data(train_val_test="train", root_data=root_data)
img_valdata_T00, img_valdata_T50, pts_valdata_T00, pts_valdata_T50 = get_img_data(train_val_test="val", root_data=root_data)
img_testdata_T00, img_testdata_T50, pts_testdata_T00, pts_testdata_T50 = get_img_data(train_val_test="test", root_data=root_data)



#%% test registration
if True:
    imgs_fixed = [itk.image_from_array(i) for i in img_testdata_T00][:5]     #end-inspiration #!last landmarks dont work, (39,3) (41,3) 
    imgs_moving = [itk.image_from_array(i) for i in img_testdata_T50][:5]

    parameter_path_base =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
    parameter_file = "Par0049.txt"

    result_image, deformation_field, result_transform_parameters = EF.registration(
        fixed_image=imgs_fixed[1], moving_image=imgs_moving[1],
        plot=True, parameter_path=parameter_path_base+parameter_file)
    # plt.imshow(result_image[:,80,:], cmap='gray')
    # plt.imshow(result_image[20:80,80,20:80], cmap='gray')
    # view (result_image)

# %% Registration of expriation to inspiration for 3 parameter files
#TODO werkt jacobanian
#TODO transformpoints op moving of fixed
#TODO parameters valideren
#TODO case 27 doet raar
def benchmark_elastix(imgs_fixed, imgs_moving, parameter_path, pts_fixed):
    # Because paths have to be specified, use this list of case names #* therefore does not work for validation atm
    cases=["04","07","09","10","26","27"]

    pts_transformed =[]
    # Register and transform fixed pts each image pair
    for i in range(len(imgs_moving)):
        # Registration
        result_image, deformation_field, result_transform_parameters = EF.registration(
                    fixed_image=imgs_fixed[i], moving_image=imgs_moving[i],
                    plot=True, parameter_path=parameter_path)

        # Transform points
        result_point_set = itk.transformix_pointset(
            imgs_moving[i], result_transform_parameters,
            fixed_point_set_file_name='./data/test/landmarks2/case_0{}/T00.txt'.format(cases[i]))
        # Select deformation and apply to fixed points to transform them
        deformations = result_point_set[:,38:41].astype(np.float32) 
        pts_transformed.append(pts_fixed[i] + deformations)
        
    return pts_transformed, result_point_set
        
def TRE(points_1, points_2):
    element_wise_difference = torch.from_numpy(points_1 - points_2) * torch.tensor(np.array([1., 1., 1.])) #element spacing np.array(imgs_fixed[0].GetSpacing())
    tre = float(torch.mean(torch.sqrt(torch.nansum(element_wise_difference ** 2, -1))))
    tre_std = float(torch.std(torch.sqrt(torch.nansum(element_wise_difference ** 2, -1))))
    # print("TRE: {} and TRE_std: {}".format(round(tre,4), round(tre_std,4)))
    return tre, tre_std


imgs_fixed = [itk.image_from_array(i) for i in img_testdata_T00]#[:2]     #end-inspiration #! [:5] last landmarks dont work, (39,3) (41,3) 
imgs_moving = [itk.image_from_array(i) for i in img_testdata_T50]#[:2]     #end-expiration
pts_fixed = pts_testdata_T00
pts_moving = pts_testdata_T50
parameter_path_base =  "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/Elastix/Parameter files/"
parameter_files = ["Par0007.txt","Par0011.txt","Par0049.txt"] #TODO test Par0049
parameter_files = ["Par0007.txt","Par0049.txt"] #TODO test Par0049
parameter_files = ["Par0007.txt"]
# parameter_files = ["Par0011.txt"]

for parameter_file in parameter_files:
    print("Elastix benchmark on test data with parameter file: {}".format(parameter_file))
    pts_transformed, result_point_set = benchmark_elastix(imgs_fixed=imgs_fixed, imgs_moving=imgs_moving, 
                                                     parameter_path=parameter_path_base+parameter_file, 
                                                     pts_fixed=pts_testdata_T00)
    
    print("TRE improvements with: {}".format(parameter_file))
    for i in range(len(pts_testdata_T00)): #!-1 bc last landmarks
        tre_initial, tre_std_initial = TRE(points_1=pts_fixed[i], 
                                        points_2=pts_moving[i])
        tre_post, tre_std_post = TRE(points_1=pts_transformed[i], #! transformed moving or fixed
                                    points_2=pts_moving[i])
        print("TRE from image pair {} imporoved from {} to {}".format(i, round(tre_initial,4), round(tre_post,4)))



#%%




















#%% write amount of points in top of file
# cases=["04","07","09","10","26","27"]
# phases=["T00","T50"]
# for case in cases:
#     for phase in phases:
#         with open('./data/test/landmarks2/case_0{}/{}.txt'.format(case,phase), 'r') as file:
#             # Read the contents of the file
#             content = file.read()
#             points = len(content.split("\n"))

#         # Open the file in write mode to overwrite its contents
#         with open('./data/test/landmarks2/case_0{}/{}.txt'.format(case,phase), 'w') as file:
#             # Write the additional line to the file
#             file.write('{}\n'.format(str(points-1)))
#             # Write the original contents of the file after the additional line
#             file.write(content)