import os

import SimpleITK as sitk
import numpy as np
import torch


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

def make_dir(path):
    parent = os.path.split(path)[0]
    parent_parent = os.path.split(parent)[0]
    if not os.path.exists(parent_parent):
        os.mkdir(parent_parent)
    if not os.path.exists(parent):
        os.mkdir(parent)
    if not os.path.exists(path):
        os.mkdir(path)

def copy_image(im_path_source, im_path_dest):
    im_sitk = sitk.ReadImage(im_path_source)
    make_dir(os.path.split(im_path_dest)[0])
    sitk.WriteImage(im_sitk, im_path_dest)


def normalize_lower_upper(im, lower, upper):
    """
    This function normalizes a numpy image according to its ... percentile

    INPUT:
    im          numpy image
    percentile  percentile to normalize with (99, 95 etc)
    """
    p0 = im.min().astype('float')
    p100 = im.max().astype('float')

    np_image_norm = im.copy()
    np_image_norm[np_image_norm < lower] = lower
    np_image_norm[np_image_norm > upper] = upper
    MAX, MIN = np_image_norm.max(), im.min()
    np_image_norm = (np_image_norm - MIN) / (MAX - MIN)
    print('Normalized from ({} , {}) to ({} , {})   using max clipping value: {}-{}'
          .format(p0, p100, np.min(np_image_norm), np.max(np_image_norm), lower, upper))
    return np_image_norm

def crop_image(im_np):
    im_np = im_np[16:-16, 16:-16, :]
    return im_np

def copy_preprocessed_image(in_path, out_path, im_type):
    im_src_sitk = sitk.ReadImage(in_path)
    im_np = sitk.GetArrayFromImage(im_src_sitk)
    im_np = crop_image(im_np)
    if im_type == 'image':
        im_np = normalize_lower_upper(im_np, im_np.min(), np.percentile(im_np, 99.5))
    im_sitk = sitk.GetImageFromArray(im_np)
    im_sitk.SetOrigin(im_src_sitk.GetOrigin())
    im_sitk.SetDirection(im_src_sitk.GetDirection())
    im_sitk.SetSpacing(im_src_sitk.GetSpacing())

    # import matplotlib.pyplot as plt
    # for slice in range(10, 160, 20):
    #     plt.figure()
    #     plt.imshow(im_np[:,:,slice], cmap='gray', vmin=0, vmax=1)
    #     plt.show()

    # write sitk image to file
    make_dir(os.path.split(out_path)[0])
    sitk.WriteImage(im_sitk, out_path)


