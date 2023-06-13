from functools import partial

import numpy as np
import torch
from monai.metrics.regression import SSIMMetric
import SimpleITK as sitk
from SimpleITK import GetArrayViewFromImage as ArrayView

def jacobian_det(displacement):
    D_x = (displacement[:, :, 1:, :-1, :-1] - displacement[:, :, :-1, :-1, :-1])
    D_y = (displacement[:, :, :-1, 1:, :-1] - displacement[:, :, :-1, :-1, :-1])
    D_z = (displacement[:, :, :-1, :-1, 1:] - displacement[:, :, :-1, :-1, :-1])

    D1 = (D_x[0, 0, ...] + 1) * ((D_y[0, 1, ...] + 1) * (D_z[0, 2, ...] + 1) - D_z[0, 1, ...] * D_y[0, 2, ...])
    D2 = (D_x[0, 1, ...]) * (D_y[0, 0, ...] * (D_z[0, 2, ...] + 1) - D_y[0, 2, ...] * D_z[0, 0, ...])
    D3 = (D_x[0, 2, ...]) * (D_y[0, 0, ...] * D_z[0, 1, ...] - (D_y[0, 1, ...] + 1) * D_z[0, 0, ...])

    det = D1 - D2 + D3
    neg = (det < 0).sum()
    folding_perc = 100 * neg / np.product(np.array(det.shape))
    jac_det_std = torch.std(det.flatten())
    return det, neg.item(), folding_perc.item(), jac_det_std.item()

def tre(points_1, points_2, elementspacing):
    element_wise_difference = (points_1.to("cuda") - points_2.to("cuda")) * torch.tensor(elementspacing.copy()).to("cuda") #!
    tre = torch.mean(torch.sqrt(torch.nansum(element_wise_difference ** 2, -1)))
    tre_std = torch.std(torch.sqrt(torch.nansum(element_wise_difference ** 2, -1)))
    return tre.item(), tre_std.item()

def multiclass_dsc(prediction, ground_truth):
    """
    :param prediction:      predicted (transformed) label (Tensor)
    :param ground_truth:    ground truth label (Tensor)
    :return:                average Dice Similarity Coefficient over the 13 classes (excluding background)
    """
    epsilon = 1E-4
    dsc_list = []
    for class_num in range(1, int(ground_truth.max()) + 1):
        # Turn the prediction and ground truth into one-hot encoded Tensors with 1's at locations with a value class_num
        pred_one_hot = (prediction == class_num).type(torch.uint8)
        gt_one_hot = (ground_truth == class_num).type(torch.uint8)
        # Flatten the Tensors
        pred_flat = pred_one_hot.view(-1)
        gt_flat = gt_one_hot.view(-1)
        # Calculate the DSC per class and add it to the list
        overlap = (pred_flat * gt_flat).sum()
        dsc = (2. * overlap + epsilon) / (pred_flat.sum() + gt_flat.sum() + epsilon)
        dsc_list.append(dsc.item())
    return np.array(dsc_list)


def multiclass_hd95(prediction, ground_truth, element_spacing):
    hausdorff_list = []
    con_sitk_1 = sitk.GetImageFromArray(prediction)
    con_sitk_1.SetSpacing(element_spacing)
    con_sitk_2 = sitk.GetImageFromArray(ground_truth)
    con_sitk_2.SetSpacing(element_spacing)

    for label in range(1, int(ground_truth.max()) + 1):
        surface_1 = sitk.LabelContour(con_sitk_1 == label, False)
        surface_2 = sitk.LabelContour(con_sitk_2 == label, False)
        distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)

        # Get distance map for contours (the distance map computes the minimum distances)
        distance_map_1 = sitk.Abs(distance_map(surface_1))
        distance_map_2 = sitk.Abs(distance_map(surface_2))

        # Find the distances to surface points of the contour.  Calculate in both directions
        one_to_2 = ArrayView(distance_map_1)[ArrayView(distance_map_2) == 1]
        two_to_1 = ArrayView(distance_map_2)[ArrayView(distance_map_1) == 1]

        # Find the 95% Distance for each direction and average
        try:
            hd_95 = (np.percentile(two_to_1, 95) + np.percentile(one_to_2, 95)) / 2.0
        except:
            hd_95 = np.nan

        hausdorff_list.append(hd_95)
    return np.array(hausdorff_list)

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict. Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            dict_obj[key] = [dict_obj[key]]  # If type is not list then make it list
        dict_obj[key].append(value)
    else:
        # As key is not in dict, add key-value pair
        dict_obj[key] = value

def sum_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Sum the value in list
        dict_obj[key] = np.nansum([dict_obj[key], value])
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value

def get_metrics_dict(dict, image_1, image_2, label_1, label_2, points_1, points_2, dvf,
                     similarity_loss, smooth_loss, gpu_speed, batch_id, dataset):
    """ LOSSES """
    loss_sim = similarity_loss.forward(image_1, image_2)
    loss_reg = smooth_loss.forward(dvf)
    loss = loss_sim + loss_reg

    """ EVALUATION METRICS """
    jac_det, neg_jac_det, folding_perc, std_jac_det = jacobian_det(dvf)
    ssim = SSIMMetric(data_range=image_1.max().unsqueeze(0)-image_1.min().unsqueeze(0), spatial_dims=3)(image_1, image_2)
    try:
        TRE, TRE_std = tre(points_1, points_2, dataset.voxel_spacing)
    except:
        TRE = np.nan
        TRE_std = np.nan

    """ APPEND STORAGE DICTIONARY """
    # Case and phase information
    case_moving, case_fixed, phase_moving, phase_fixed = dataset.get_case_info(batch_id)
    for key, value in zip(['batch_id', 'case_moving', 'case_fixed',
                           'phase_moving', 'phase_fixed'],
                          [batch_id, case_moving, case_fixed,
                           phase_moving, phase_fixed]):
        append_value(dict, key, value)

    # Losses and metrics
    for key, value in zip(['loss', 'loss_sim', 'loss_reg',
                           'ssim', 'tre', 'tre_std', 'folding_perc', 'std_jac_det', 'gpu_speed'],
                          [loss.item(), loss_sim.item(), loss_reg.item(),
                           ssim.item(), TRE, TRE_std, folding_perc, std_jac_det, gpu_speed]):
        append_value(dict, key, value)

    # Dices and HD

    if label_1!=None:
        if label_1.sum().item() > 0 and label_2.sum().item() > 0:
            dices = multiclass_dsc(label_1, label_2)
            hd_95 = multiclass_hd95(label_1.cpu().squeeze().numpy(),
                                label_2.cpu().squeeze().numpy(),
                                dataset.voxel_spacing)

            for key, value in zip(dataset.organ_list, dices):
                append_value(dict, 'DSC_' + key, value)
            append_value(dict, 'DSC_mean', np.nanmean(dices))
            append_value(dict, 'DSC_30', np.nanpercentile(dices, 30))
            for key, value in zip(dataset.organ_list, hd_95):
                append_value(dict, 'HD_' + key, value)
            append_value(dict, 'HD_mean', np.nanmean(hd_95))
    return dict, (loss_sim, loss_reg, loss)