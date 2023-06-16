import torch.utils.data
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
import argparse

import sys
# from transformers import SpatialTransformer #!
# from utils import set_seed, read_pts   #!

import torch.nn as nn
import torch.nn.functional as F

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

def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

def read_pts(file_name, skiprows=0):
    return torch.tensor(np.loadtxt(file_name, skiprows=skiprows), dtype=torch.float32)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', fractional=False):
        super().__init__()

        self.mode = mode
        self.fractional = fractional
        # create sampling grid
        if fractional:
            vectors = [torch.arange(0, s) / (s - 1) for s in size]
        else:
            vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            if self.fractional:
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] - 0.5)
            else:
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class Dataset(torch.utils.data.Dataset):
    """
    GENERAL DATASET
    """
    def __init__(self, train_val_test, augmenter=None, augment=None, save_augmented=False, plot=False):
        self.overfit = False
        self.train_val_test = train_val_test
        self.augmenter = augmenter
        self.augment = augment
        self.save_augmented = save_augmented
        self.plot = plot

    def adjust_shape(self, multiple_of=16):
        new_shape = tuple([int(np.ceil(shp / multiple_of) * multiple_of) for shp in self.inshape])
        self.offsets = [shp - old_shp for (shp, old_shp) in zip(new_shape, self.inshape)]
        self.inshape = new_shape

    @staticmethod
    def get_image_header(path):
        image = sitk.ReadImage(path)
        dim = np.array(image.GetSize())
        voxel_sp = np.array(image.GetSpacing())
        return dim[::-1], voxel_sp[::-1]

    def read_image_sitk(self, path):
        return sitk.ReadImage(path)

    def read_image_np(self, path):
        image = sitk.ReadImage(path)
        image_np = sitk.GetArrayFromImage(image).astype('float32')
        return image_np

    def get_paths(self, i):
        """Get the path to images and labels"""
        # Load in the moving- and fixed image/label
        moving_path, fixed_path = self.moving_img[i], self.fixed_img[i]
        return moving_path, fixed_path

    def overfit_one(self, i):
        self.overfit = True
        self.moving_img, self.fixed_img = [self.moving_img[i]], [self.fixed_img[i]]

    def subset(self, rand_indices):
        temp = [self.moving_img[i] for i in rand_indices]
        self.moving_img = temp
        temp = [self.fixed_img[i] for i in rand_indices]
        self.fixed_img = temp
        temp = [self.moving_lbl[i] for i in rand_indices]
        self.moving_lbl = temp
        temp = [self.fixed_lbl[i] for i in rand_indices]
        self.fixed_lbl = temp

    def __len__(self):
        return len(self.fixed_img)

    def save_nifti(self, img, fname):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        img_sitk = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_sitk, fileName=fname)

    def __getitem__(self, i):
        """Load the image/label into a tensor"""

        # Get image paths and load images
        moving_path, fixed_path = self.get_paths(i)
        moving_np = self.read_image_np(moving_path)
        fixed_np = self.read_image_np(fixed_path)

        # Transform the arrays into tensors and add an extra dimension for the "channels"
        moving_t = torch.from_numpy(moving_np).unsqueeze(0)
        fixed_t = torch.from_numpy(fixed_np).unsqueeze(0)


        # Generate DVFs on the fly and apply to original moving image
        if self.augment=="gryds":
            DVF_augment, DVF_respiratory = self.augmenter.generate_on_the_fly(moving_t)
            moving_t_new = self.augmenter.transformer(src=(moving_t).to("cuda").unsqueeze(0), flow=DVF_augment).squeeze(0)
            DVF_composed, _ = self.augmenter.composed_transform(DVF_augment, DVF_respiratory)
            fixed_t_new = self.augmenter.transformer(src=(moving_t).to("cuda").unsqueeze(0), flow=DVF_composed).squeeze(0)
            moving_t = moving_t_new
            fixed_t = fixed_t_new

            if self.save_augmented:
                # Create directories
                if not os.path.exists(os.path.split(moving_path.replace('image', 'synthetic_dvf_aug'))[0]):
                    os.makedirs(os.path.split(moving_path.replace('image', 'synthetic_dvf_aug'))[0])
                if not os.path.exists(os.path.split(moving_path.replace('image', 'synthetic_dvf_resp'))[0]):
                    os.makedirs(os.path.split(moving_path.replace('image', 'synthetic_dvf_resp'))[0])
                if not os.path.exists(os.path.split(moving_path.replace('image', 'synthetic_image'))[0]):
                    os.makedirs(os.path.split(moving_path.replace('image', 'synthetic_image'))[0])

                # Save all niftis
                self.save_nifti(DVF_augment, moving_path.replace('image', 'synthetic_dvf_aug'))
                self.save_nifti(DVF_respiratory,moving_path.replace('image', 'synthetic_dvf_resp'))
                self.save_nifti(moving_t, moving_path.replace('image', 'synthetic_image'))
                self.save_nifti(fixed_t, fixed_path.replace('image', 'synthetic_image'))
        
            return moving_t, fixed_t
        
        elif self.augment=="SMOD":
            img_artificial_inhaled = self.augmenter.generate_img_artificial(img_base=self.augmenter.imgs_inhaled_to_atlas[i],
                                                                        DVF_components=self.augmenter.DVF_inhaled_components, 
                                                                        sigma=self.augmenter.sigma1, breathing=False)
            
            img_artificial_exhaled = self.augmenter.generate_img_artificial(img_base=img_artificial_inhaled,
                                                                        DVF_components=self.augmenter.DVF_exhaled_components, 
                                                                        sigma=self.augmenter.sigma2, breathing=True)
            

            if self.plot:
                self.augmenter.plot_data_augmpairs(img_artificial_inhaled, img_artificial_exhaled, title="Artificial inhaled and exhaled images")
            
            # change 3d np.arrays to 4d torch
            img_artificial_inhaled = torch.from_numpy(img_artificial_inhaled).unsqueeze(0)
            img_artificial_exhaled = torch.from_numpy(img_artificial_exhaled).unsqueeze(0)
            
            return img_artificial_inhaled, img_artificial_exhaled
            
        else:
            return moving_t, fixed_t
            
class DatasetLung(Dataset):
    def __init__(self, train_val_test, root_data, augmenter=None, augment=None, save_augmented=False, version="2.1D", phases='in_ex'):
        super().__init__(train_val_test, augmenter, augment, save_augmented)
        self.set = 'lung'
        self.extension = '.nii.gz'
        self.root_data = root_data
        self.version = version
        self.phases = phases
        self.organ_list = []
        # self.img_folder = f'{root_data}/LUNG_CT/V{version}_PREPROCESSED/{train_val_test}/image/***'
        # self.landmarks_folder = f'{root_data}/LUNG_CT/V{version}_PREPROCESSED/{train_val_test}/landmarks/***'
        self.img_folder = f'{root_data}/{train_val_test}/image/***'
        self.landmarks_folder = f'{root_data}/{train_val_test}/landmarks/***'
        self.init_paths()
        self.inshape, self.voxel_spacing = self.get_image_header(self.fixed_img[0])
        # self.inshape = np.array((160,128,160))

    def init_paths(self):
        if self.phases == 'in_ex':
            self.phases_fixed = [0, 90]
            self.phases_moving = [50, 50]
            if self.version == '2.1D' or self.version == '2.1E' or self.version == '2.1E':
                self.phases_fixed = [0]
                self.phases_moving = [50]
        elif self.phases == 'all':
            self.phases_fixed = []
            self.phases_moving = []
            for i in range(0, 100, 10):
                for j in range(0, 100, 10):
                    if i is not j:
                        self.phases_fixed.append(i)
                        self.phases_moving.append(j)

        if self.train_val_test != 'train':
            self.phases_fixed = [0]
            self.phases_moving = [50]

        # Get all file names inside the data folder
        self.img_paths, self.landmarks_paths = glob(self.img_folder), glob(self.landmarks_folder)
        self.img_paths.sort()
        self.landmarks_paths.sort()
        self.fixed_img, self.moving_img = [], []
        self.fixed_pts, self.moving_pts = [], []
        for img_folder in self.img_paths:
            landmark_folder = img_folder.replace('image', 'landmarks')
            for phase_fixed, phase_moving in zip(self.phases_fixed, self.phases_moving):
                f = os.path.join(img_folder, 'T{:02d}{}'.format(phase_fixed, self.extension))
                m = os.path.join(img_folder, 'T{:02d}{}'.format(phase_moving, self.extension))
                fl = os.path.join(landmark_folder, 'T{:02d}.txt'.format(phase_fixed))
                ml= os.path.join(landmark_folder, 'T{:02d}.txt'.format(phase_moving))
                if os.path.exists(f) and os.path.exists(m):
                    self.fixed_img.append(f)
                    self.moving_img.append(m)
                    if os.path.exists(fl) and os.path.exists(ml):
                        self.fixed_pts.append(fl)
                        self.moving_pts.append(ml)
                    else:
                        self.fixed_pts.append('')
                        self.moving_pts.append('')


    def get_case_info(self, i):
        moving_path, fixed_path = self.get_paths(i)
        case_m = int(moving_path[-13:-11])
        case_f = int(fixed_path[-13:-11])
        phase_m = int(moving_path[-9:-7])
        phase_f = int(fixed_path[-9:-7])
        return case_m, case_f, phase_m, phase_f

    def get_landmarks(self, i):
        fixed_landmarks = read_pts(self.fixed_pts[i]) + torch.tensor(self.offsets)
        moving_landmarks = read_pts(self.moving_pts[i]) + torch.tensor(self.offsets)

        indices_all = []
        for pts in [fixed_landmarks, moving_landmarks]:
            indices_all = indices_all + np.argwhere(pts[:, 0] >= self.inshape[0]).tolist()[0]
            indices_all = indices_all + np.argwhere(pts[:, 1] >= self.inshape[1]).tolist()[0]
            indices_all = indices_all + np.argwhere(pts[:, 2] >= self.inshape[2]).tolist()[0]
            indices_all = indices_all + np.argwhere(pts[:, 0] < 0).tolist()[0]
            indices_all = indices_all + np.argwhere(pts[:, 1] < 0).tolist()[0]
            indices_all = indices_all + np.argwhere(pts[:, 2] < 0).tolist()[0]

        indices_all = np.unique(indices_all)

        if len(indices_all) > 0:
            fixed_landmarks = np.delete(fixed_landmarks, indices_all, axis=0)
            moving_landmarks = np.delete(moving_landmarks, indices_all, axis=0)
        return moving_landmarks, fixed_landmarks

    def overfit_one(self, i):
        self.overfit = True
        self.moving_img, self.fixed_img = [self.moving_img[i]], [self.fixed_img[i]]
        self.moving_pts, self.fixed_pts = [self.moving_pts[i]], [self.fixed_pts[i]]

class GrydsPhysicsInformed():
    def __init__(self, **kwargs):
        # self.args = args
        self.max_deform_base = {'augment': kwargs.pop('max_deform_aug', 3.2),
                               'resp_coarse': kwargs.pop('max_deform_coarse', 3.2),
                               'resp_fine': kwargs.pop('max_deform_fine', 3.2)}
        self.grid_size = {'augment': kwargs.pop('grid_size_aug', [2] * 3),
                          'resp_coarse': kwargs.pop('grid_size_coarse', [4] * 3),
                          'resp_fine': kwargs.pop('grid_size_fine', [8] * 3)}

    def generate_bspline_params_augment(self, grid_size, max_deform_base, zyx_factor=[1, 1, 1]):
        # The values are random floats between -theta and theta, so the deformations are random over the domain.
        # Larger theta results in larger displacements. Larger grid spacings result in a shift from local to global
        # deformations.
        # Relation between mesh_size and grid_size:
        # mesh_size = [x // gs for x, gs in zip(img_shape, grid_spacing)]
        # grid_size = [int(g + 1) for g in mesh_size]
        zyx_max_deform = [max_deform_base*fctr for fctr in zyx_factor]
        bspline_parameters = np.zeros(grid_size + [3])
        for i, value in enumerate(zyx_max_deform):
            bspline_parameters[:, :, :, i] = np.random.uniform(-value, value, grid_size)
        return bspline_parameters

    def generate_bspline_params_respiratory(self, grid_size, grid_type, max_deform_base=3.2, zyx_factor=[1, 1, 1]):
        # Set default transforms
        ranges = [min_max * np.ones(grid_size + [3]) for min_max in [-1., 1.]]
        for dim, factor in enumerate(zyx_factor):
            for pos in range(2):
                ranges[pos][:,:,:,dim] = max_deform_base * factor * ranges[pos][:,:,:,dim]

        mids = [grid_size[dim] // 2 for dim in range(3)]

        if grid_type == 'coarse':
                # Top half
                ranges[0][:mids[0], :, :, 0] = -3 * max_deform_base
                ranges[1][:mids[0], :, :, 0] = max_deform_base
                # Bottom half
                ranges[0][mids[0]:, :, :, 0] = -4 * max_deform_base
                ranges[1][mids[0]:, :, :, 0] = 0

        # Fix borders to zero displacement
        if grid_type == 'fine':
            # most posterior plane, fix left-right and posterior-anterior direction
            ranges[0][:, -1, :, 1:] = 0
            ranges[1][:, -1, :, 1:] = 0

            # most cranial plane, fix left-right and posterior-anterior direction
            ranges[0][0, :, :, 1:] = 0
            ranges[1][0, :, :, 1:] = 0

            # most left and right plane, fix left-right and posterior-anterior direction
            ranges[0][:, :, 0, 1:] = 0
            ranges[1][:, :, 0, 1:] = 0
            ranges[0][:, :, -1, 1:] = 0
            ranges[1][:, :, -1, 1:] = 0

        bspline_parameters = np.random.uniform(ranges[0], ranges[1], grid_size + [3])
        return bspline_parameters

    def generate_DVF_from_bspline_param(self, bspline_params, grid_size, img_shape):
        # We create a reference image in the shape that we need because we are returning our deformation vector field.
        # If we only need our resulting warped image, we can also pass the input image and directly return the result.
        ref_image = sitk.Image(img_shape.tolist(), sitk.sitkFloat32)
        mesh_size = [int(s - 1) for s in grid_size]

        # Create a bspline transformation initializer with the reference image and the mesh size.
        bspline_transform = sitk.BSplineTransformInitializer(ref_image, mesh_size)

        # Initialize the shift in the control points. The mesh size is equal to the number of control points - spline order.
        new_shape = [shp + 3 for shp in mesh_size]
        new_shape = new_shape + [3]
        new_params = np.array(bspline_transform.GetParameters()).reshape(new_shape, order='F')
        new_params[1:-1, 1:-1, 1:-1, :] = bspline_params
        bspline_transform.SetParameters(new_params.flatten(order='F').tolist())

        # Transform the transformation into a displacement vector field.
        # Note that SimpleITK works with backward transforms, so the displacement field shows the displacements from
        # result to input (every voxels contains a vector that shows where that voxel CAME from, not where it is going).
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(ref_image)
        displacement_filter.SetOutputPixelType(sitk.sitkVectorFloat64)
        dvf = displacement_filter.Execute(bspline_transform)

        dvf_np = sitk.GetArrayFromImage(dvf).astype('float32')
        dvf_t = torch.from_numpy(dvf_np)
        dvf_t = dvf_t.permute(3, 2, 1, 0).unsqueeze(0)
        return dvf_t.to("cuda")

    def generate_DVF(self, augment_or_respiratory):
        '''
        This function generates synthetic displacement vector fields (DVFs) and
        deformed images and returns them.
        '''
        # assert self.dims == 3, "The model is only implemented for 3D data" #!
        # img_shape = self.inshape 
        img_shape = np.array((160,128,160))#!

        if augment_or_respiratory == 'augment':
            bspline_params = self.generate_bspline_params_augment(grid_size=self.grid_size[augment_or_respiratory],
                                                          max_deform_base=self.max_deform_base[augment_or_respiratory],
                                                          zyx_factor=[4, 2, 1])
            DVF = self.generate_DVF_from_bspline_param(bspline_params=bspline_params, #!
                                                       grid_size=self.grid_size[augment_or_respiratory],
                                                       img_shape=img_shape)
        elif augment_or_respiratory == 'respiratory':

            bspline_params_coarse = self.generate_bspline_params_respiratory(grid_size=self.grid_size['resp_coarse'], #!
                                                                    grid_type ='coarse',
                                                                    # experiment = self.args.experiment, #!
                                                                    max_deform_base=self.max_deform_base['resp_coarse'], #!
                                                                    zyx_factor = [4, 2, 1])

            bspline_params_fine = self.generate_bspline_params_respiratory(grid_size=self.grid_size['resp_fine'], #!
                                                                  grid_type = 'fine',
                                                                #   experiment = self.args.experiment, #!
                                                                  max_deform_base=self.max_deform_base['resp_fine'], #!
                                                                  zyx_factor = [1, 1, 1])
                                                                #   zero_borders=self.args.zero_borders) #!

            DVF_coarse = self.generate_DVF_from_bspline_param(bspline_params=bspline_params_coarse,
                                                              grid_size=self.grid_size['resp_coarse'], #!
                                                              img_shape=img_shape)
            DVF_fine = self.generate_DVF_from_bspline_param(bspline_params=bspline_params_fine,
                                                            grid_size=self.grid_size['resp_fine'], #!
                                                            img_shape=img_shape)
            DVF, DVF_coarse_transformed = self.composed_transform(DVF_coarse, DVF_fine)
        return DVF

class Augmentation_gryds(GrydsPhysicsInformed):
    def __init__(self, sig_noise=0.005):
        super().__init__()
        # self.args = args
        self.sig_noise = sig_noise
        # self.transformer = SpatialTransformer(self.inshape).to(self.args.device)
        self.transformer = SpatialTransformer(np.array((160,128,160))).to("cuda") #!

    def save_nifti(self, img, fname):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        img_sitk = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_sitk, fileName=fname)

    def add_gaussian_noise(self, img):
        noise = torch.from_numpy(np.random.normal(0, self.sig_noise, img.shape)).type(img.dtype).to("cuda")
        return img + noise

    def rescale_DVF(self, DVF, reverse=False):
        """

        Parameters
        ----------
        DVF : torch tensor
            containing the displacements in voxels

        Returns
        -------
        rescaled DVF : torch tensor
            Containing rescaled displacement between -1 and 1

        reverse : bool (optional)
            When true the rescaling is reversed. From [-1, 1] to voxels.

        """
        delta = 2. / (torch.tensor(DVF.shape[:-1], dtype=DVF.dtype, device=DVF.device) - 1)
        if reverse:
            delta = 1. / delta
        return DVF * delta

    def generate_on_the_fly(self, img):
        # DVF_augment = self.generate_DVF(img, 'augment')
        # DVF_respiratory = self.generate_DVF(img, 'respiratory')
        DVF_augment = self.generate_DVF('augment') #!
        DVF_respiratory = self.generate_DVF('respiratory')
        return DVF_augment, DVF_respiratory

    def composed_transform(self, DVF1, DVF2):
        # Use this transformed grid to interpolated DVF1 (first transform)
        DVF1_transformed = self.transformer(src=DVF1, flow=DVF2)
        DVF = DVF1_transformed + DVF2
        return DVF, DVF1_transformed


class Augmentation_SMOD():
    def __init__(self, root_data, original_dataset, sigma1, sigma2, plot=False, 
                 load_atlas=True, load_preprocessing=False, save_preprocessing=False):
        self.root_data = root_data
        # image generation parameters
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.plot = plot                #plot registration and transformation steps
        self.load_atlas = load_atlas    #wheather to generate atlas or load existing one
        # Load train data
        self.imgs_inhaled, self.imgs_exhaled = zip(*[(itk.image_from_array(img_inhaled[0]), itk.image_from_array(img_exhaled[0])) 
                                             for img_inhaled, img_exhaled in original_dataset])
        if load_preprocessing==False:
            # Generate DVF components for inhaled augmentation
            self.DVF_inhaled_components, self.imgs_inhaled_to_atlas, self.DVFs_to_atlas = self.preprocessing_inhaled()
            # Generate DVF components for exhaled augmentation
            self.DVF_exhaled_components = self.preprocessing_exhaled()
        
        # Performing the preprocessing steps are always the same and slow
        # Therefore a function to load the data once created    
            if save_preprocessing==True:
                print("Saving preprocessing data...")
                if not os.path.exists(self.root_data+'/preprocessing/'):
                    os.makedirs(self.root_data+'/preprocessing/')
                # Save the list of NumPy arrays to a file
                preprocessing=[]
                preprocessing.append(self.DVF_inhaled_components)
                preprocessing.append(self.DVF_exhaled_components)
                np.save(self.root_data+'/preprocessing/DVF_components.npy', preprocessing)
                

                img_template = nib.load(root_data+'/train/image/case_001/T00.nii.gz')
                for i, image in enumerate(self.imgs_inhaled_to_atlas):
                    img_nib = nib.Nifti1Image(image, img_template.affine)
                    nib.save(img_nib, self.root_data+'/preprocessing/image_to_atlas{}.nii.gz'.format(i))
    
        else:
            # Load the list of NumPy arrays from the file
            print("Loading preprocessing data...")
            preprocessing = np.load(self.root_data+'/preprocessing/DVF_components.npy', allow_pickle=True)
            self.DVF_inhaled_components = preprocessing[0]
            self.DVF_exhaled_components = preprocessing[1]
            
            self.imgs_inhaled_to_atlas = []
            for i in range(len(self.imgs_inhaled)):
                image = nib.load(self.root_data+'/preprocessing/image_to_atlas{}.nii.gz'.format(i))
                # image = itk.image_from_array(ndi.rotate((image).astype(np.float32),0))
                image = itk.image_from_array(ndi.rotate(image.get_fdata(),0))
                self.imgs_inhaled_to_atlas.append(image)
        
        print("Done with augmentation initialization")

    # Basic Elastix and Gryds (library) functions
    def registration(self, fixed_image, moving_image, method="rigid", 
                    parameter_path=None, output_directory=""):
        """Function that calls Elastix registration function
        Args:
            fixed_image: list with arrays of images (same as moving_image)
            method: string with either one of the standard registration methods
                or the name of a parameter file
        """

        # Define parameter object
        parameter_object = itk.ParameterObject.New()
        if parameter_path != None:
            parameter_object.AddParameterFile(parameter_path)    #e.g. Par007.txt
        else:
            parameter_object.SetParameterMap(parameter_object.GetDefaultParameterMap(method))

        # Registration
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object, #number_of_threads=8, 
            log_to_console=False, output_directory=output_directory)

        # Deformation field
        deformation_field = itk.transformix_deformation_field(moving_image, result_transform_parameters)
        # print("Registration step complete")

        # Jacobanian #? does not work always 
        # jacobians = itk.transformix_jacobian(moving_image, result_transform_parameters) #!
        # # Casting tuple to two numpy matrices for further calculations.
        # spatial_jacobian = np.asarray(jacobians[0]).astype(np.float32)
        # det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float32)
        # print("Number of foldings in transformation:",np.sum(det_spatial_jacobian < 0))
        
        if self.plot:
            self.plot_registration(fixed_image, moving_image, result_image, deformation_field, full=True)

        return result_image, deformation_field, result_transform_parameters

    def DVF_conversion(self, DVF_itk): 
        """Converts DVF from itk to DVF usable with gryds
        Note that gryds is here the library used and not the alternte augmenation method gryds*"""
        # Reshapre DVF from (160, 128, 160, 3) to (3, 160, 128, 160)
        reshaped_DVF = np.transpose(np.asarray(DVF_itk), (3, 0, 1, 2))  
        
        # DVF elastix in pixels while DVF gryds in proportions. Therefore scale each direction with their pixelsize. 
        # DVF[0] works medial-lateral on 160 pixels, DVF[2] cranial-caudial on 160 pixels, DVF[1] posterior-anterior in 128 pixels
        DVF_scaled = np.asarray([reshaped_DVF[0]/160, reshaped_DVF[1]/128, reshaped_DVF[2]/160])
        
        # gryds transformation takes DVFs in different order: CC, AP, ML
        DVF_gryds = [DVF_scaled[2], DVF_scaled[1], DVF_scaled[0]]

        return DVF_gryds

    def transform(self, DVF_itk, img_moving):
        """Transform an image with a DVF using the gryds library
        Since the gryds format for DVFs is different to ITK, first the DVF is scaled
        Args:
            DVF_itk: DVF as array or itk object with displacements in pixels
                (converted to displacments scaled to image size with DVF_conversion)
            img_moving: to transform image
        """
        DVF_gryds = self.DVF_conversion(DVF_itk=DVF_itk)

        bspline_transformation = gryds.BSplineTransformation(DVF_gryds)
        an_image_interpolator = gryds.Interpolator(img_moving)
        img_deformed = an_image_interpolator.transform(bspline_transformation)
        if self.plot:
            self.plot_registration(fixed_image=img_moving, moving_image=img_deformed, deformation_field=DVF_itk, full=True, 
                                result_image=np.asarray(img_moving) - np.asarray(img_deformed), title="transformation with averaged atlas DVF",
                                name1="train image", name2="result image", name3="subtracted image", name4="average atlas DVF")
        return img_deformed

    def plot_registration(self, fixed_image, moving_image, result_image, deformation_field,
            name1="fixed image", name2="moving image", name3="result image", name4="deformation field",
            title="In- and output of registration - frontal/transverse/saggital" , full=True):
        """Plot fixed, moving result image and deformation field
        Called after registration to see result"""
        
        if full==False:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(fixed_image[:, 64, :])
            axs[0, 0].set_title(name1)
            axs[0, 1].imshow(moving_image[:, 64, :])
            axs[0, 1].set_title(name2)
            axs[1, 0].imshow(result_image[:, 64, :])
            axs[1, 0].set_title(name3)
            axs[1, 1].imshow(deformation_field[:, 64, :, 2])
            axs[1, 1].set_title(name4)
            fig.suptitle("In- and output of registration - frontal")

        elif full==True:
            fig, axs = plt.subplots(2, 6, figsize=(30, 10), dpi=80) #, dpi=80
            plt.rc('font', size=20) 
            axs[0, 0].imshow(fixed_image[:, 64, :])
            axs[0, 0].set_title(name1)
            axs[0, 1].imshow(moving_image[:, 64, :])
            axs[0, 1].set_title(name2)
            axs[1, 0].imshow(result_image[:, 64, :])
            axs[1, 0].set_title(name3)
            axs[1, 1].imshow(deformation_field[:, 64, :, 2])
            axs[1, 1].set_title(name4)
            
            axs[0, 2].imshow(fixed_image[64, :, :])
            axs[0, 2].set_title(name1)
            axs[0, 3].imshow(moving_image[64, :, :])
            axs[0, 3].set_title(name2)
            axs[1, 2].imshow(result_image[64, :, :])
            axs[1, 2].set_title(name3)
            axs[1, 3].imshow(deformation_field[64, :, :, 0])
            axs[1, 3].set_title(name4)
            
            axs[0, 4].imshow(fixed_image[:, :, 64])
            axs[0, 4].set_title(name1)
            axs[0, 5].imshow(moving_image[:, :, 64])
            axs[0, 5].set_title(name2)
            axs[1, 4].imshow(result_image[:, :, 64])
            axs[1, 4].set_title(name3)
            axs[1, 5].imshow(deformation_field[:, :, 64, 1])
            axs[1, 5].set_title(name4)
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def validate_DVFs(self, DVFs, DVFs_inverse, img_moving):
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
            img_ogDVF = self.transform(DVF_itk=DVFs[i], img_moving=img_moving, plot=False)
            ax[i,1].imshow(img_ogDVF[:,64,:])
            ax[i,1].set_title("original deformed with DVF")
            # DVFinverse and img+DVFinverse
            ax[i,2].imshow(np.asarray(DVFs_inverse[i])[:,64,:,2])
            img_invDVF = self.transform(DVF_itk=DVFs_inverse[i], img_moving=img_moving, plot=False)
            ax[i,3].imshow(img_invDVF[:,64,:])
            ax[i,3].set_title("original deformed with DVFinv")
            # img+DVFinverse+DVF
            img_3 = self.transform(DVF_itk=DVFs[i], img_moving=img_invDVF, plot=False)
            ax[i,4].imshow(img_3[:,64,:])
            ax[i,4].set_title("original deformed with DVFinf and DVF")
        plt.tight_layout()
        plt.show()

    # Atlas generation
    def generate_atlas(self, img_data):
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
            
            result_image, deformation_field, _ = self.registration(
                fixed_image=A0, moving_image=moving_image, 
                method="rigid")
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
                
                result_image, deformation_field, _ = self.registration(
                    fixed_image=An, moving_image=registered_set[i], 
                    method="affine")  # ?Affine
                re_registered_set.append(result_image)

            A_new_array = sum(np.asarray(re_registered_set)) / len(np.asarray(re_registered_set)) 
            A_new = itk.image_from_array(A_new_array.astype(np.float32))
            
            if self.plot:
                self.plot_registration(An, A_new, np.asarray(An) - np.asarray(A_new), deformation_field, full=False)
            
            iteration += 1

            # Calculate L2 norm and break when convergence is reached
            L2 = np.linalg.norm(np.asarray(An) - np.asarray(A_new)) 
            print("L2: {}   dL2: {}".format(L2, L2_last - L2))
            
            if (L2_last - L2) <= 1:
                break
        
        print("Atlas generation complete/n")
        return An

    def register_to_atlas(self, method="affine", inverse=True):
        """Generate DVFs from set images registered on atlas image"""
        params_path = self.root_data.replace("data","transform_parameters")
        if not os.path.exists(params_path):
            os.makedirs(params_path)

        DVFs_list, DVFs_inverse_list, imgs_to_atlas = [], [], []

        for i in range(len(self.imgs_inhaled)):
            result_image, DVF, result_transform_parameters = self.registration(
                fixed_image=self.img_atlas, moving_image=self.imgs_inhaled[i], 
                method=method, output_directory=params_path)
            DVFs_list.append(DVF)
            imgs_to_atlas.append(result_image)
            
            # Inverse DVF
            if inverse:
                parameter_object = itk.ParameterObject.New()
                parameter_map = parameter_object.GetDefaultParameterMap(method)
                parameter_map['HowToCombineTransforms'] = ['Compose']
                parameter_object.AddParameterMap(parameter_map)
                
                inverse_image, inverse_transform_parameters = itk.elastix_registration_method(
                    self.imgs_inhaled[i], self.imgs_inhaled[i],
                    parameter_object=parameter_object,
                    initial_transform_parameter_file_name=os.path.join(params_path+"/TransformParameters.0.txt"))
                
                inverse_transform_parameters.SetParameter(
                    0, "InitialTransformParametersFileName", "NoInitialTransform")
                DVF_inverse = itk.transformix_deformation_field(self.imgs_inhaled[i], inverse_transform_parameters)
                DVFs_inverse_list.append(DVF_inverse)
            
            
        return DVFs_list, DVFs_inverse_list, imgs_to_atlas

    # DVF generation
    def dimreduction(self, DVFs):
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

    def generate_artificial_DVFs(self, DVFs_artificial_components, sigma):
        """Generate artificial DVFs from list with to_atlas_DVFs
        Args:
            DVFs_artificial_components (list with arrays): DVF_mean, DVF_Ud from dimreduction()
            sigma (int): random scaling component 100: visual deformations, 500: too much
            DVFs (list): list with DVF's either in shape (160, 128, 160, 3) or itk.itkImagePython.itkImageVF33
        Returns:
            DVFs_artificial (list with arrays): artificial DVFs with shape (160, 128, 160, 3)
        """
        # Unpack mean and PCA components from dimreduction()
        DVF_mean, DVF_Ud = DVFs_artificial_components
        
        DVF_artificial = np.zeros((DVF_mean[0].shape[0], 3))
        for j in range(3):
            x = np.random.normal(loc=0, scale=sigma, size=DVF_Ud[j].shape[0])
            # Paper: vg = Vmean + U*x*d 
            DVF_artificial[:,j] = DVF_mean[j] + np.dot(DVF_Ud[j].T, x)  #(3276800,1)
        DVF_artificial=np.reshape(DVF_artificial, (160, 128, 160, 3))

        if self.plot:
            img_grid = np.zeros((160, 128, 160))
            line_interval = 5  # Adjust this value to change the interval between lines
            img_grid[:, ::line_interval, :] = 1
            img_grid[:, :, ::line_interval] = 1
            img_grid[::line_interval, :, :] = 1
            self.transform(DVF_artificial, img_grid)
            
        return DVF_artificial


    # MAIN functions  
    def preprocessing_inhaled(self):
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
        print("Generating atlas image")
        if self.load_atlas==True:
            self.img_atlas = self.generate_atlas(img_data=self.imgs_inhaled)
        else:
            # Load already generated atlas image
            img_atlas = nib.load(self.root_data+'atlas/atlasv1.nii.gz')
            img_atlas = img_atlas.get_fdata()
            self.img_atlas = itk.image_from_array(ndi.rotate((img_atlas).astype(np.float32),0)) # itk.itkImagePython.itkImageF3
            if self.plot:
                plt.imshow(img_atlas[:,64,:])
                plt.title("Pregenerated atlas image")

        
        # (2) Register to atlas for DVFinverse and imgs_to_atlas  
        print("Regestration of inhaled images to atlas")  
        DVFs, DVFs_inverse, imgs_to_atlas = self.register_to_atlas()
        
        # Optionally validate DVFs and DVFs_inverse (uncomment next line)
        # self.validate_DVFs(DVFs, DVFs_inverse, img_moving=imgs_inhaled[0])
        
        # (3) Get components needed for artificial DVF generation
        print("Generating artificial DVF components")
        DVF_inhaled_components = self.dimreduction(DVFs=DVFs_inverse)
        
        return DVF_inhaled_components, imgs_to_atlas, DVFs

    def preprocessing_exhaled(self):
        # register inhaled to exhaled with bspline to get the breathing motion
        DVFs_breathing=[]
        for i in range(len(self.imgs_inhaled)):
            result_image, DVF_breathing, result_transform_parameters = self.registration(
                fixed_image=self.imgs_exhaled[i], moving_image=self.imgs_inhaled[i], 
                method="bspline")#, parameter_path=parameter_path_base+parameter_file)
            DVFs_breathing.append(DVF_breathing)

        # generate artificial DVFs that model breathing motion from inhaled to exhaled
        print("Generating artificial DVFs - breathing motion")
        DVF_exhaled_components = self.dimreduction(DVFs=DVFs_breathing)
        return DVF_exhaled_components

    # Phase generation
    def generate_img_artificial(self, img_base, DVF_components, sigma, breathing=False):
        print("Generating artificial DVF")
        DVF_artificial = self.generate_artificial_DVFs(DVFs_artificial_components=DVF_components, 
                                                       sigma=sigma)
    
        # generate artificial images
        print("Generating artificial image")
        img_artificial = self.transform(img_moving=img_base, DVF_itk=DVF_artificial)
        
        # if self.plot:
        #     if breathing==False:
        #         self.plot_registration(fixed_image=self.imgs_inhaled, moving_image=img_base, result_image=img_artificial, deformation_field=DVF_artificial,
        #                             name1="Original inhaled", name2="Inhaled to atlas", name3="Artificial inhaled", name4="artificial DVF",
        #                             title="Creating artificial images", full=True)
        #     elif breathing==True:
        #         self.plot_registration(fixed_image=self.imgs_exhaled, moving_image=img_base, result_image=img_artificial, deformation_field=DVF_artificial,
        #                             name1="Original exhaled", name2="Artificial inhaled", name3="Artificial exhaled", name4="artificial DVF",
        #                             title="Creating artificial images", full=True)

        return img_artificial

    def plot_data_augmpairs(img_artificial_inhaled, img_artificial_exhaled, title):
        fig, axes = plt.subplots(len(img_artificial_inhaled), 3, figsize=(3*2.5, len(img_artificial_inhaled)*3))        
        fig.suptitle(title, y=1)  
        for i in range(len(img_artificial_inhaled)):
            axes[i,0].imshow(img_artificial_inhaled[i][:,64,:])
            axes[i,1].imshow(img_artificial_exhaled[i][:,64,:])
            img_neg=np.asarray(img_artificial_exhaled[i]) - np.asarray(img_artificial_inhaled[i])
            axes[i,2].imshow(img_neg[:,64,:])
            axes[i,0].axis('off')
            axes[i,1].axis('off')
            axes[i,2].axis('off') 
        plt.tight_layout()    
        plt.plot()

    # def write_augmented_data(path, foldername, imgs_inhaled, imgs_exhaled):
    #     img = nib.load(path+'train/image/case_001/T00.nii.gz')
    #     for i in range(len(imgs_inhaled)):
    #         folder_path = os.path.join(path,foldername,"image", str(i).zfill(3))
    #         if not os.path.exists(folder_path):
    #             os.makedirs(folder_path)
                
    #         img_nib = nib.Nifti1Image(imgs_inhaled[i], img.affine)
    #         nib.save(img_nib, os.path.join(folder_path, 'T00.nii.gz'))
    #         img_nib = nib.Nifti1Image(imgs_exhaled[i], img.affine)
    #         nib.save(img_nib, os.path.join(folder_path, 'T50.nii.gz'))
    


if __name__ == '__main__':
    root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'

    # Original data
    dataset_original = DatasetLung(train_val_test='train', version='',
                                   root_data=root_data, augmenter=None, phases='in_ex')
    imgs_inhaled, imgs_exhaled = zip(*[(img_inhaled, img_exhaled) for img_inhaled, img_exhaled in dataset_original])

    #TODO: add non varying data 
    # Gryds
    augmenter_gryds = Augmentation_gryds()
    dataset_synthetic_gryds = DatasetLung(train_val_test='train', version='', root_data=root_data, 
                                    augmenter=augmenter_gryds, augment="gryds", save_augmented=False, phases='in_ex')
    imgs_inhaled_gryds, imgs_exhaled_gryds = zip(*[(img_inhaled, img_exhaled) for img_inhaled, img_exhaled in dataset_synthetic_gryds])

    # SMOD
    augmenter_SMOD = Augmentation_SMOD(root_data=root_data, original_dataset=dataset_original,
                                    sigma1=15000, sigma2=1500, plot=False, load_atlas=True, 
                                    load_preprocessing=True, save_preprocessing=False)
    dataset_synthetic_SMOD = DatasetLung(train_val_test='train', version='', root_data=root_data, 
                                    augmenter=augmenter_SMOD, augment="SMOD", save_augmented=False, phases='in_ex')
    imgs_inhaled_SMOD, imgs_exhaled_SMOD = zip(*[(img_inhaled, img_exhaled) for img_inhaled, img_exhaled in dataset_synthetic_SMOD])
    

    for i in range(len(dataset_original)):
        fig, axs = plt.subplots(3, 3, figsize=(20,20))
        axs[0,0].imshow(np.asarray(imgs_inhaled[i][0])[:,64,:], cmap='gray')
        axs[0,0].set_title('inhaled', fontsize=40)
        axs[0, 1].imshow(np.asarray(imgs_exhaled[i][0])[:,64,:], cmap='gray')
        axs[0, 1].set_title('exhaled', fontsize=40)
        axs[0, 2].imshow((np.asarray(imgs_inhaled[i][0])-np.asarray(imgs_exhaled[i][0]))[:,64,:], cmap='gray')
        axs[0, 2].set_title('inhales-exhaled', fontsize=40)
        
        axs[1, 0].imshow(np.asarray(imgs_inhaled_gryds[i][0].to("cpu"))[:,64,:], cmap='gray')
        axs[1, 0].set_title('inhaled gryds', fontsize=40)
        axs[1, 1].imshow(np.asarray(imgs_exhaled_gryds[i][0].to("cpu"))[:,64,:], cmap='gray')
        axs[1, 1].set_title('exhaled gryds', fontsize=40)
        axs[1, 2].imshow((np.asarray(imgs_inhaled_gryds[i][0].to("cpu")) - np.asarray(imgs_exhaled_gryds[i][0].to("cpu")))[:,64,:], cmap='gray')
        axs[1, 2].set_title('inhaled-exhaled gryds', fontsize=40)
        
        axs[2, 0].imshow(np.asarray(imgs_inhaled_SMOD[i][0].to("cpu"))[:,64,:], cmap='gray')
        axs[2, 0].set_title('inhaled SMOD', fontsize=40)
        axs[2, 1].imshow(np.asarray(imgs_exhaled_SMOD[i][0].to("cpu"))[:,64,:], cmap='gray')
        axs[2, 1].set_title('exhaled SMOD', fontsize=40)
        axs[2, 2].imshow((np.asarray(imgs_inhaled_SMOD[i][0].to("cpu") - np.asarray(imgs_exhaled_SMOD[i][0].to("cpu"))))[:,64,:], cmap='gray')
        axs[2, 2].set_title('inhales-exhaled SMOD', fontsize=40)
        for ax in axs.flatten():
            ax.set_axis_off()
        plt.tight_layout()
        fig.show()


    