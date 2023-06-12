import argparse
import numpy as np
import pandas as pd
import torch
from monai.losses import GlobalMutualInformationLoss

from datasets.datasets import DatasetLung
from executor.losses import Grad, NCC
from executor.train_val import validate_epoch, train_epoch
from model.utils import save_model, init_model#, set_level_sequential_training_2
from utils.neptune import init_neptune
from utils.utils import set_seed

torch.backends.cudnn.benchmark = True  # speed ups

# import os
# os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100"
# print(os.environ.get("PYDEVD_WARN_EVALUATION_TIMEOUT"))

base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/"
base_path = "C:/Users/Quinten Vroemen/Documents/MV_codespace/BEP_MIA_DIR/"

""" ARGUMENT PARSER """
parser = argparse.ArgumentParser(description='J01_VIT - train script')
parser.add_argument('-net', '--network', type=str, metavar='', default='msvit', help='network architecture used')
parser.add_argument('--root_checkpoints', type=str, metavar='',
                    default=base_path+"multi-step-ViT/checkpoints", help='')  #default='/home/bme001/20210003/projects/J01_VIT/checkpoints'
parser.add_argument('--root_output', type=str, metavar='',
                    default=base_path+'multi-step-ViT/output/model_val_metrics', help='')     # default='/home/bme001/20210003/projects/J01_VIT/output/model_val_metrics'
parser.add_argument('--root_data', type=str, metavar='', default=base_path+"4DCT/data", help='')  # default='/home/bme001/20210003/data'
parser.add_argument('-seed', '--random_seed', type=int, metavar='', default=1000, help='random seed')
parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda', help='device / gpu used')
parser.add_argument('-nept', '--mode_neptune', type=str, metavar='', default='async',
                    help='Neptune run mode: async | debug')
# parser.add_argument('-run_nr', type=str, metavar='', default='1') #!added


# MultiStepViT architecture
parser.add_argument('--vit_steps', type=int, metavar='', default=2, help='the number of steps of the MultiStepViT')
parser.add_argument('--patch_size', type=int, metavar='', default=[8, 4], nargs='+', help='patch size')
parser.add_argument('--stages', type=int, metavar='', default=[3, 4], nargs='+', help='nr. of stages')
parser.add_argument('--embed_dim', type=int, metavar='', default=[48, 48], nargs='+', help='embedded dimensions') #! [48]
parser.add_argument('--depths', type=int, metavar='', default=[2,2], nargs='+', help='nr. of MSA blocks')   #! [2]
parser.add_argument('--num_heads', type=int, metavar='', default=[4,4], nargs='+',                          #! [4]
                    help='nr. of attention heads in the MSA block')
parser.add_argument('--window_size', type=int, metavar='', default=[2,2], nargs='+', help='window size')    #! [2]

# Hyper-parameters for training
parser.add_argument('-loss', '--similarity_loss', type=str, metavar='', default='ncc',
                    help='similarity loss | nmi | ncc ')
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=1e-4, help='learning rate')
parser.add_argument("-rw", '--reg_weight', type=float, metavar='', default=1, help='regularization (smoothing) weight')
parser.add_argument('-ep', '--epochs', type=int, metavar='', default=50, help='nr of epochs you want to train on')
parser.add_argument('-bs', '--batch_size', type=int, metavar='', default=1,
                    help='batch size you want to use during training')

# Dataset
parser.add_argument('-set', '--dataset', type=str, metavar='', default='lung', help='dataset')
# parser.add_argument('-aug', '--augmentation', type=str, metavar='', default='none') 
parser.add_argument('-aug', '--augmentation', type=str, metavar='', default='SMOD') 
# parser.add_argument('augmentation', choices=['none', 'SMOD', 'gryds'])  #!
parser.add_argument('-v', '--version', type=str, metavar='', default='', help='preprocessing version')
parser.add_argument('--overfit', action='store_true', help='overfit on 1 image during training')
args = parser.parse_args()
print(vars(args))
set_seed(args.random_seed)

if __name__ == "__main__":
    """ CONFIG NEPTUNE """
    args.mode = 'train'
    # If you uncomment the code below and make an account on Neptune you can monitor the progress of your training
    # Also uncomment the lines starting with "run"  --> for example line 75 "run["dataset/size"] = len(train_dataset)" to actually log something to neptune
    run, args, epoch = init_neptune(args)
    # run = None # comment this line and the line below if you uncommented the lines above
    # epoch = 0

    """ CONFIG DATASET """
    if args.dataset == 'lung':
        if args.augmentation == 'none':
            train_dataset = DatasetLung('train', root_data=args.root_data, version=args.version)
            val_dataset = DatasetLung('val', root_data=args.root_data, version=args.version)
        elif args.augmentation == 'SMOD':
            train_dataset = DatasetLung('train', folder_augment="artificial_N5_S10000_1000", root_data=args.root_data, version=args.version)
            val_dataset = DatasetLung('val', root_data=args.root_data, version=args.version)
    print("Training dataset size: ", len(train_dataset))
    train_dataset.adjust_shape(multiple_of=32)
    val_dataset.adjust_shape(multiple_of=32)
    if args.overfit:
        train_dataset.overfit_one(i=0)
        val_dataset = train_dataset
        val_dataset.train_val_test = 'val'
    run["dataset/size"] = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    """ INITIALIZE MODEL """
    model = init_model(args, img_size=train_dataset.inshape)
    run["model/trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["model/architecture"] = model
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.mode_neptune != 'debug':
        save_model(model, args, epoch, run)

    """INITIALIZE LOSS FUNCTIONS """
    if args.similarity_loss == 'ncc':
        similarity_loss = NCC(args.device)
    if args.similarity_loss == 'nmi':
        similarity_loss = GlobalMutualInformationLoss()
    smooth_loss = Grad(penalty='l2', loss_mult=args.reg_weight)

    """ TRAINING """
    print('\n----- Training -----')
    # Baseline losses and metrics before training
    validate_epoch(model, train_loader, run, args,
                   similarity_loss, smooth_loss)
    validate_epoch(model, val_loader, run, args,
                   similarity_loss, smooth_loss)

    # Train the model for the specified amount of epochs
    epoch += 1
    while epoch < args.epochs + 1:
        print(f'\n[epoch {epoch} / {args.epochs}]')
        # Train and validate for one epoch
        train_epoch(model, train_loader, optimizer, run, args,
                    similarity_loss, smooth_loss)
        metrics = validate_epoch(model, val_loader, run, args,
                       similarity_loss, smooth_loss)

        # Save the model each epoch
        epoch += 1
        if args.mode_neptune != 'debug' and epoch % 5 == 0:
            save_model(model, args, epoch, run)

    df = pd.DataFrame(metrics)
    csv_path = '{}/csv/{}_{}_ep-{:04d}.csv'.format(args.root_output, args.run_nr, args.network, epoch - 1)
    df.to_csv(csv_path)
    df.to_pickle(csv_path.replace('csv', 'pkl'))
    run.stop()
