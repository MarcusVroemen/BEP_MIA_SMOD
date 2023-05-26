"""
Training/validation functions
"""
import sys
from time import perf_counter

import torch
from tqdm import tqdm

from executor.metrics import get_metrics_dict
from utils.neptune import log_dict_2_neptune
from torch.cuda.amp import GradScaler

MIXED_PRECISION = True
def train_epoch(model, data_loader, optimizer, run, args,
                similarity_loss, smooth_loss):
    """
    Train for one epoch
    """
    # Creates a GradScaler once at the beginning of training.
    if MIXED_PRECISION:
        scaler = GradScaler(enabled=True)

    total_loss = 0
    model.train()
    for batch_idx, (img_moving, img_fixed) in enumerate(tqdm(data_loader, file=sys.stdout)):
        # Take the img_moving and fixed images to the GPU
        img_moving, img_fixed = img_moving.to(args.device), img_fixed.to(args.device)

        # Zero out the optimizer gradients as these are normally accumulated
        optimizer.zero_grad()

        """ Loss calculation """
        # Get the transformed img_moving image and the corresponding Displacement Vector Field
        dvf, img_warped = model(img_moving, img_fixed)

        # Compute the similarity loss between the transformed img_moving image and the fixed image (ignoring padding)
        loss_sim = similarity_loss.forward(img_fixed, img_warped)
        loss_reg = smooth_loss.forward(dvf)
        loss = loss_sim + loss_reg

        # Update the total loss
        total_loss += loss.item()

        # Back propagate loss and update parameters with optimizer, retain_graph=True allows doing backpropagation twice
        if MIXED_PRECISION:
            scaler.scale(loss_sim).backward(retain_graph=True)
            scaler.scale(loss_reg).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        """ Save iteration to Neptune """
        # run["train/its/loss"].log(loss.item())
        # run["train/its/loss_reg"].log(loss_reg.item())

    """ Save epoch to Neptune """
    # run["train/epochs/loss"].log(total_loss / len(data_loader))
    return total_loss / len(data_loader)


def validate_epoch(model, data_loader, run, args,
                   similarity_loss, smooth_loss):
    """
    validate
    """
    # Initialize empty dictionaries
    storage = dict()
    model.eval()
    with torch.no_grad():
        for batch_id, (img_moving, img_fixed) in \
                enumerate(tqdm(data_loader, file=sys.stdout)):
            # Take the source and target images to the GPU and add batch dimension
            # After that, get the transformed source image and the corresponding Displacement Vector Field
            st = perf_counter()
            img_moving, img_fixed = img_moving.to(args.device), img_fixed.to(args.device)
            dvf, img_warped = model(img_moving, img_fixed)
            et = perf_counter()
            gpu_speed = et - st

            # Landmarks
            try:
                pts_moving, pts_fixed = data_loader.dataset.get_landmarks(batch_id)
                pts_fixed_def = model.stl.warp_landmarks(pts_fixed, dvf)
            except:
                pts_moving, pts_fixed, pts_fixed_def = None, None, None

            # Labels
            try:
                lbl_moving, lbl_fixed = data_loader.dataset.get_labels(batch_id)
                lbl_moving, lbl_fixed = lbl_moving.to(args.device).unsqueeze(0), lbl_fixed.to(args.device).unsqueeze(0)
                lbl_warped = model.stl_binary(lbl_moving, dvf)
            except:
                lbl_moving, lbl_fixed, lbl_warped = None, None, None

            # Calculate losses and validation metrics (Dice, TRE etc.) and put in dictionary
            storage, _ = get_metrics_dict(dict=storage,
                                          image_1=img_fixed, image_2=img_warped,
                                          label_1=lbl_fixed, label_2=lbl_warped,
                                          points_1=pts_moving, points_2=pts_fixed_def, dvf=dvf,
                                          similarity_loss=similarity_loss,
                                          smooth_loss=smooth_loss,
                                          gpu_speed=gpu_speed, batch_id=batch_id,
                                          dataset=data_loader.dataset)

    # save metrics to neptune
    # try:
    #     log_dict_2_neptune(run=run, dict=storage, prefix=f'{data_loader.dataset.train_val_test}/epochs')
    # except:
    #     pass
    return storage
