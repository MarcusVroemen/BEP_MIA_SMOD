import os

import numpy as np
import torch
import SimpleITK as sitk

def to_cpu(img):
    img = img.squeeze().detach().cpu()
    return img

def norm_percentile(img, p):
    """Normalize an image (Numpy array) based on the 1st and 99th percentile and clip to a range of [0, 1]"""
    img = np.clip(img, np.percentile(img, 100-p), np.percentile(img, p))
    return img

def save_model(model, args, epoch, run):
    if args.save_model:
        model_path = '../model_checkpoints/{}/{}_{}_ep-{:03d}.pt'.format(args.run_nr, args.run_nr, args.network, epoch)
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, model_path)
        run["model/path"] = model_path

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


def save_torch_to_image(im, voxel_sp, image_path):
    # make sitk image
    im = im.numpy()
    im_sitk = sitk.GetImageFromArray(im)
    im_sitk.SetSpacing(voxel_sp)

    # write sitk image to file
    make_dir(os.path.split(image_path)[0])
    sitk.WriteImage(im_sitk, image_path)



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
