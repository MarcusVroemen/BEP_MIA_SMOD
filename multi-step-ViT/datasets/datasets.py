import os
import random
from glob import glob

import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from datasets.utils import copy_preprocessed_image, pad

def read_pts(file_name, skiprows=0):
    return torch.tensor(np.loadtxt(file_name, skiprows=skiprows), dtype=torch.float32)


class Dataset(torch.utils.data.Dataset):
    """
    GENERAL DATASET
    """
    def __init__(self, train_val_test):
        self.overfit = False
        self.train_val_test = train_val_test

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

    def __getitem__(self, i):
        # Get image paths and load images
        moving_path, fixed_path = self.get_paths(i)
        moving_np = self.read_image_np(moving_path)
        fixed_np = self.read_image_np(fixed_path)

        # Padding
        moving_np, _ = pad(moving_np, self.inshape, 0)
        fixed_np, _ = pad(fixed_np, self.inshape, 0)
        # moving_np = moving_np[16:-16,16:-16,:]
        # fixed_np = fixed_np[16:-16,16:-16,:]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(moving_np[72,:,:], cmap='gray')
        # plt.show()

        # Transform the arrays into tensors and add an extra dimension for the "channels"
        moving_t = torch.from_numpy(moving_np).unsqueeze(0)
        fixed_t = torch.from_numpy(fixed_np).unsqueeze(0)
        return moving_t, fixed_t

class DatasetLung(Dataset):
    def __init__(self, train_val_test, version, root_data, folder_augment=None, phases='in_ex'):
        super().__init__(train_val_test)
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
        if folder_augment!=None:
            self.img_folder_augment = f'{root_data}/{folder_augment}/image/***'
            print(self.img_folder_augment)
        else:
            self.img_folder_augment = None
        self.init_paths()
        self.inshape, self.voxel_spacing = self.get_image_header(self.fixed_img[0])
        self.offsets = [0, 0, 0] #!

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

        if self.train_val_test != ('train' or 'artificial'):
            self.phases_fixed = [0]
            self.phases_moving = [50]

        # Get all file names inside the data folder

        if self.img_folder_augment!=None:
            self.img_paths, self.landmarks_paths = glob(self.img_folder)+glob(self.img_folder_augment), glob(self.landmarks_folder)
        else:
            self.img_paths, self.landmarks_paths = glob(self.img_folder), glob(self.landmarks_folder)
        # print(self.img_paths)
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
