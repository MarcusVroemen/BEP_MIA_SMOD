import os
import sys
from glob import glob

import torch
import torch.nn.functional as F

import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk


def pad(array, shape, pad_value):
    """
    Pads an array with the given padding value to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...
    padded = pad_value * np.ones(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array
    return padded, slices


def read_pts(file_name, skiprows=0):
    return torch.tensor(np.loadtxt(file_name, skiprows=skiprows), dtype=torch.float32)


def identity_grid(shape, unity=False):
   if unity:
       vectors = [torch.arange(0, s) / (s - 1) for s in shape]
   else:
       vectors = [torch.arange(0, s) for s in shape]
   grids = torch.meshgrid(vectors)
   grid = torch.stack(grids)
   grid = torch.unsqueeze(grid, 0)
   grid = grid.type(torch.FloatTensor)
   return grid


class SpatialTransformer(torch.nn.Module):
   """
   N-D Spatial Transformer
   Obtained from https://github.com/voxelmorph/voxelmorph
   """


   def __init__(self, shape, mode='bilinear', unity=False):
       super().__init__()
       self.shape = shape
       self.mode = mode
       self.unity = unity
       grid = identity_grid(shape=shape, unity=unity)
       self.register_buffer('grid', grid)
       # self.grid = identity_grid(shape=shape, unity=unity).to(device)
       # registering the grid as a buffer cleanly moves it to the GPU, but it also
       # adds it to the state dict. this is annoying since everything in the state dict
       # is included when saving weights to disk, so the model files are way bigger
       # than they need to be. so far, there does not appear to be an elegant solution.
       # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict


   def forward(self, src, flow):
       # new locations
       new_locs = self.grid + flow
       shape = flow.shape[2:]


       # need to normalize grid values to [-1, 1] for resampler
       if self.unity:
           for i in range(len(shape)):
               new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] - 0.5)
       else:
           for i in range(len(shape)):
               new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)


       # move channels dim to last position
       # also not sure why, but the channels need to be reversed
       if len(shape) == 2:
           new_locs = new_locs.permute(0, 2, 3, 1)
           new_locs = new_locs[..., [1, 0]]
       elif len(shape) == 3:
           new_locs = new_locs.permute(0, 2, 3, 4, 1)
           new_locs = new_locs.flip([-1])
       return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


   def warp_landmarks(self, src, flow):
       # move channels dim to last position
       # also not sure why, but the channels need to be reversed
       flow = flow.permute(0, 2, 3, 4, 1)
       idx = torch.round(src).to(torch.int64)
       displacements = flow[:, idx[:, 0].tolist(), idx[:, 1].tolist(), idx[:, 2].tolist(), :].squeeze()
       if self.unity:
           displacements = displacements * torch.Tensor(data=flow.shape[1:4]).to(flow.device)
       points_warped = src + displacements
       return points_warped




class Dataset(torch.utils.data.Dataset):
   """
   GENERAL DATASET
   """
   def __init__(self, train_val_test, augment_def, max_deform_base, sig_noise=0.005, device='cpu'):
       self.overfit = False
       self.train_val_test = train_val_test
       self.augment_def = augment_def
       self.grid_size = [5] * 3  # meaning that control points are 5x5x5 > mesh size = 2x2x2
       self.zyx_factor = [1] * 3 #
       self.spline_order = 3
       self.max_deform_base = max_deform_base
       self.sig_noise = sig_noise
       self.device = device

   def adjust_shape(self, multiple_of=16):
       old_shape, _ = self.get_image_header(self.fixed_img[0])
       new_shape = tuple([int(np.ceil(shp / multiple_of) * multiple_of) for shp in old_shape])
       self.inshape = new_shape
       self.offsets = [shp - old_shp for (shp, old_shp) in zip(new_shape, old_shape)]


   @staticmethod
   def get_image_header(path):
       image = sitk.ReadImage(path)
       dim = np.array(image.GetSize())
       voxel_sp = np.array(image.GetSpacing())
       return dim[::-1], voxel_sp[::-1]


   def read_image_sitk(self, path):
       if os.path.exists(path):
           image = sitk.ReadImage(path)
       else:
           image_np = np.zeros(self.inshape, dtype='float32')
           image = sitk.GetImageFromArray(image_np)
       return image


   def read_image_np(self, path):
       if os.path.exists(path):
           image = sitk.ReadImage(path)
           image_np = sitk.GetArrayFromImage(image).astype('float32')
       else:
           image_np = np.zeros(self.inshape, dtype='float32')
       return image_np


   def get_paths(self, i):
       """Get the path to images and labels"""
       # Load in the moving- and fixed image/label
       moving_path, fixed_path = self.moving_img[i], self.fixed_img[i]
       moving_label_path, fixed_label_path = self.moving_lbl[i], self.fixed_lbl[i]
       return moving_path, fixed_path, moving_label_path, fixed_label_path


   def overfit_one(self, i):
       self.overfit = True
       self.moving_img, self.fixed_img = [self.moving_img[i]], [self.fixed_img[i]]
       self.moving_lbl, self.fixed_lbl = [self.moving_lbl[i]], [self.fixed_lbl[i]]
       self.identity_transforms = []


   def subset(self, rand_indices):
       temp = [self.moving_img[i] for i in rand_indices]
       self.moving_img = temp
       temp = [self.fixed_img[i] for i in rand_indices]
       self.fixed_img = temp
       temp = [self.moving_lbl[i] for i in rand_indices]
       self.moving_lbl = temp
       temp = [self.fixed_lbl[i] for i in rand_indices]
       self.fixed_lbl = temp


   def init_resamplers(self):
       self.resampler = sitk.ResampleImageFilter()
       self.resampler.SetInterpolator(sitk.sitkLinear)
       self.resampler.SetDefaultPixelValue(0)


       self.resampler_bin = sitk.ResampleImageFilter()
       self.resampler_bin.SetInterpolator(sitk.sitkNearestNeighbor)
       self.resampler_bin.SetDefaultPixelValue(0)


   def generate_bspline_params(self, grid_size, max_deform_base, zyx_factor=[1, 1, 1]):
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


   def generate_DVF_from_bspline_param(self, bspline_params, grid_size, img_shape):
       # We create a reference image in the shape that we need because we are returning our deformation vector field.
       # If we only need our resulting warped image, we can also pass the input image and directly return the result.
       ref_image = sitk.Image(img_shape.tolist(), sitk.sitkFloat32)
       mesh_size = [int(s - 1) for s in grid_size]


       # Create a bspline transformation initializer with the reference image and the mesh size.
       bspline_transform = sitk.BSplineTransformInitializer(ref_image, mesh_size)


       # Initialize the shift in the control points. The mesh size is equal to the number of control points - spline order.
       new_shape = [shp + self.spline_order for shp in mesh_size]
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
       return dvf_t.to(self.device)


   def augment_on_the_fly(self, image, label):
       bspline_params = self.generate_bspline_params(grid_size=self.grid_size,
                                                     max_deform_base=self.max_deform_base,
                                                     zyx_factor=self.zyx_factor)
       DVF = self.generate_DVF_from_bspline_param(bspline_params=bspline_params,
                                                  grid_size=self.grid_size,
                                                  img_shape=self.inshape)
       augmented_image = self.transformer(src=image.unsqueeze(0), flow=DVF)
       augmented_label = self.transformer_binary(src=label.unsqueeze(0), flow=DVF)
       return augmented_image.squeeze(0), augmented_label.squeeze(0)


   def add_gaussian_noise(self, img):
       noise = torch.from_numpy(np.random.normal(0, self.sig_noise, img.shape)).type(img.dtype).to(self.device)
       return img + noise


   def __len__(self):
       return len(self.fixed_img)


   def __getitem__(self, i):
       """Load the image/label into a tensor"""
       # get paths and images
       moving_path, fixed_path, moving_label_path, fixed_label_path = self.get_paths(i)
       moving_np, fixed_np = self.read_image_np(moving_path), self.read_image_np(fixed_path)
       moving_label_np, fixed_label_np = self.read_image_np(moving_label_path), self.read_image_np(fixed_label_path)


       # Padding
       moving_np, _ = pad(moving_np, self.inshape, 0)
       fixed_np, _ = pad(fixed_np, self.inshape, 0)
       moving_label_np, _ = pad(moving_label_np, self.inshape, 0)
       fixed_label_np, _ = pad(fixed_label_np, self.inshape, 0)


       # Transform the arrays into tensors and add an extra dimension for the "channels"
       moving_t = torch.from_numpy(moving_np).unsqueeze(0)
       fixed_t = torch.from_numpy(fixed_np).unsqueeze(0)
       moving_label_t = torch.from_numpy(moving_label_np).unsqueeze(0)
       fixed_label_t = torch.from_numpy(fixed_label_np).unsqueeze(0)
       
       plot=False
       if plot:
           fig, axs = plt.subplots(2, 2)
           axs[0, 0].imshow(moving_t[:, :, 64, :].squeeze().numpy(), cmap='gray')
           axs[0, 0].set_title('moving')
           axs[0, 1].imshow(fixed_t[:, :, 64, :].squeeze().numpy(), cmap='gray')
           axs[0, 1].set_title('fixed')

       if self.train_val_test == 'train' and self.augment_def:
           # Augmentation
           moving_t, moving_label_t = self.augment_on_the_fly(moving_t, moving_label_t)
           fixed_t, fixed_label_t = self.augment_on_the_fly(fixed_t, fixed_label_t)
           moving_t = self.add_gaussian_noise(moving_t)
           fixed_t = self.add_gaussian_noise(fixed_t)
           
           if plot:
               axs[1, 0].imshow(moving_t[:, :, 64, :].squeeze().numpy(), cmap='gray')
               axs[1, 0].set_title('moving augmented')
               axs[1, 1].imshow(fixed_t[:, :, 64, :].squeeze().numpy(), cmap='gray')
               axs[1, 1].set_title('fixed augmented')
               plt.show()

       return moving_t, fixed_t, moving_label_t, fixed_label_t


class DatasetLung(Dataset):
    def __init__(self, train_val_test, root_data, folder_augment=None,
                    augment_def=False, max_deform_base=20, sig_noise=0.005,
                    phases='in_ex', device='cpu'):
        super().__init__(train_val_test, augment_def, max_deform_base, sig_noise, device)
        self.set = 'lung'
        self.extension = '.nii.gz'
        self.phases = phases
        if train_val_test=="artificial":
            self.img_folder = f'{root_data}/{train_val_test}/{folder_augment}/image/***'
            self.landmarks_folder = ""
        else:
            self.img_folder = f'{root_data}/{train_val_test}/image/***'
            self.landmarks_folder = f'{root_data}/{train_val_test}/landmarks/***'
        self.init_paths()
        self.init_resamplers()
        self.inshape, self.voxel_spacing = self.get_image_header(self.fixed_img[0])
        self.transformer = SpatialTransformer(self.inshape).to(self.device)
        self.transformer_binary = SpatialTransformer(shape=self.inshape, mode='nearest').to(self.device)
        self.offsets = [0, 0, 0]

    def init_paths(self):
        if self.phases == 'in_ex':
            self.phases_fixed = [0, 90]
            self.phases_moving = [50, 50]

        if self.train_val_test != 'train':
            self.phases_fixed = [0]
            self.phases_moving = [50]

        # Get all file names inside the data folder
        self.img_paths, self.landmarks_paths = glob(self.img_folder), glob(self.landmarks_folder)
        self.img_paths.sort(), self.landmarks_paths.sort()
        self.fixed_img, self.moving_img = [], []
        self.fixed_lbl, self.moving_lbl = [], []
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
                    self.fixed_lbl.append('')
                    self.moving_lbl.append('')
                    if os.path.exists(fl) and os.path.exists(ml):
                        self.fixed_pts.append(fl)
                        self.moving_pts.append(ml)
                    else:
                        self.fixed_pts.append('')
                        self.moving_pts.append('')


    def get_case_sessions(self, i):
        moving_path, fixed_path, _, _ = self.get_paths(i)
        case = int(fixed_path[-13:-11])
        ses_m = int(moving_path[-9:-7])
        ses_f = int(fixed_path[-9:-7])
        return case, case, ses_m, ses_f

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
        self.moving_lbl, self.fixed_lbl = [self.moving_lbl[i]], [self.fixed_lbl[i]]
        self.moving_pts, self.fixed_pts = [self.moving_pts[i]], [self.fixed_pts[i]]


if __name__ == '__main__':
    device = 'cpu'
    root_data = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/4DCT/data/'
    train_dataset = DatasetLung('train', root_data=root_data, augment_def=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    for batch_idx, (img_moving, img_fixed, lbl_moving, lbl_fixed) in enumerate(tqdm(train_loader, file=sys.stdout)):
        # Take the img_moving and fixed images to the GPU
        img_moving, img_fixed = img_moving.to(device), img_fixed.to(device)
