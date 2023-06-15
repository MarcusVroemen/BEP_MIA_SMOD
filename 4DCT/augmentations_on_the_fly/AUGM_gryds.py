import torch.utils.data
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
import argparse

import sys
from transformers import SpatialTransformer
from utils import set_seed, read_pts

import torch.nn as nn
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    """
    GENERAL DATASET
    """
    def __init__(self, train_val_test, augmenter=None, save_augmented=False):
        self.overfit = False
        self.train_val_test = train_val_test
        self.augmenter = augmenter
        self.save_augmented = save_augmented
        if self.augmenter is not None:
            self.augment = True
        else:
            self.augment = False

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
        # Get image paths and load images
        moving_path, fixed_path = self.get_paths(i)
        moving_np = self.read_image_np(moving_path)
        fixed_np = self.read_image_np(fixed_path)

        # Transform the arrays into tensors and add an extra dimension for the "channels"
        moving_t = torch.from_numpy(moving_np).unsqueeze(0)
        fixed_t = torch.from_numpy(fixed_np).unsqueeze(0)

        # Generate DVFs on the fly and apply to original moving image
        if self.augment:
            DVF_augment, DVF_respiratory = self.augmenter.generate_on_the_fly(moving_t)
            moving_t_new = self.augmenter.transformer(src=moving_t.unsqueeze(0), flow=DVF_augment).squeeze(0)
            DVF_composed, _ = self.augmenter.composed_transform(DVF_augment, DVF_respiratory)
            fixed_t_new = self.augmenter.transformer(src=moving_t.unsqueeze(0), flow=DVF_composed).squeeze(0)
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


class DatasetLung(Dataset):
    def __init__(self, train_val_test, version, root_data, augmenter=None, save_augmented=False, phases='in_ex'):
        super().__init__(train_val_test, augmenter, save_augmented)
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
    def __init__(self, args, **kwargs):
        self.args = args
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
        return dvf_t.to(self.args.device)

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


class Augmentation(GrydsPhysicsInformed):
    def __init__(self, args, sig_noise=0.005):
        super().__init__(args)
        self.args = args
        self.sig_noise = sig_noise
        # self.transformer = SpatialTransformer(self.inshape).to(self.args.device)
        self.transformer = SpatialTransformer(np.array((160,128,160))).to(self.args.device) #!

    def save_nifti(self, img, fname):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        img_sitk = sitk.GetImageFromArray(img)
        sitk.WriteImage(img_sitk, fileName=fname)

    def add_gaussian_noise(self, img):
        noise = torch.from_numpy(np.random.normal(0, self.sig_noise, img.shape)).type(img.dtype).to(self.args.device)
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


if __name__ == '__main__':
    root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_value', type=int, default=1000)
    parser.add_argument('--dataroot', type=str, default=root_data)
    parser.add_argument('--augment_type', type=str, default='GrydsPhysicsInformed',
                        help="should be GrydsPhysicsInformed")
    parser.add_argument('--phase', type=str, default='train', help='train, val, test')
    parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda:0', help='device / gpu used')
    args = parser.parse_args()
    set_seed(args.seed_value)

    # example of original dataset without augmentation
    dataset_original = DatasetLung(train_val_test='train', version='',
                                   root_data=args.dataroot, augmenter=None, phases='in_ex')
    # img_data_T00, img_data_T50, img_data_T90 = prepara_traindata(root_data=root_data)
    # moving, fixed = img_data_T00[0], img_data_T50[0]
    moving, fixed = dataset_original[0]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(moving[0,:,64,:], cmap='gray')
    axs[1].imshow(fixed[0,:,64,:], cmap='gray')
    fig.show()

    # example of synthetic data (augmented)
    augmenter = Augmentation(args)
    dataset_synthetic = DatasetLung(train_val_test='train', version='',
                                   root_data=args.dataroot, augmenter=augmenter, save_augmented=True, phases='in_ex')
    # moving_synth, fixed_synth = dataset_original[0]
    for i in range(len(dataset_original)):
        moving_synth, fixed_synth = dataset_synthetic[i]
        moving_synth = moving_synth.to("cpu")
        fixed_synth = fixed_synth.to("cpu")

        fig, axs = plt.subplots(2, 2)
        axs[0,0].imshow(moving[0,:,64,:], cmap='gray')
        axs[0,0].set_title('moving')
        axs[0, 1].imshow(fixed[0,:,64,:], cmap='gray')
        axs[0, 1].set_title('fixed')
        axs[1, 0].imshow(moving_synth[0,:,64,:], cmap='gray')
        axs[1, 0].set_title('moving_synth')
        axs[1, 1].imshow(fixed_synth[0,:,64,:], cmap='gray')
        axs[1, 1].set_title('fixed_synth')
        fig.show()


