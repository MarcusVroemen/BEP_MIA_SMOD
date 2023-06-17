import argparse

import numpy as np
import pandas as pd
import torch
from monai.losses import GlobalMutualInformationLoss

from datasets.datasets import DatasetLung
from executor.losses import Grad, NCC
from executor.train_val import validate_epoch
from model.utils import save_model, init_model
from utils.neptune import re_init_neptune
# TODO:

torch.backends.cudnn.benchmark = True  # speed ups

base_path = "/home/bme001/20203531/BEP/BEP_MIA_DIR/BEP_MIA_DIR/"
# base_path = "C:/Users/Quinten Vroemen/Documents/MV_codespace/BEP_MIA_DIR/"
# base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/"

""" ARGUMENT PARSER """
parser = argparse.ArgumentParser(description='J01_VIT - validation script')
parser.add_argument('-run', '--run_nr', type=str, metavar='', default='1',help='')
parser.add_argument('-ep', '--epochs', type=int, metavar='', default=11, help='nr of epochs you want to evaluate on')
parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda', help='device / gpu used')
parser.add_argument('--root_output', type=str, metavar='', 
                    default=base_path+'multi-step-ViT/output/model_val_metrics', help='') #default='/home/bme001/20210003/projects/J01_VIT/output/model_val_metrics'
parser.add_argument('-set', '--dataset', type=str, metavar='', default='lung', help='dataset')

# MultiStepViT architecture (used in all experiments)
parser.add_argument('--vit_steps', type=int, metavar='', default=2, help='the number of steps of the MultiStepViT')
parser.add_argument('--patch_size', type=int, metavar='', default=[8, 4], nargs='+', help='patch size')
parser.add_argument('--stages', type=int, metavar='', default=[3, 4], nargs='+', help='nr. of stages')
parser.add_argument('--embed_dim', type=int, metavar='', default=[48, 96], nargs='+', help='embedded dimensions') 
parser.add_argument('--depths', type=int, metavar='', default=[2,2], nargs='+', help='nr. of MSA blocks')   
parser.add_argument('--num_heads', type=int, metavar='', default=[4,4], nargs='+',                          
                    help='nr. of attention heads in the MSA block')
parser.add_argument('--window_size', type=int, metavar='', default=[2,2], nargs='+', help='window size') 

args = parser.parse_args()
print(vars(args))

if __name__ == "__main__":
    """ CONFIG NEPTUNE """
    args.mode = 'test' #!
    run, args, epoch = re_init_neptune(args)

    """ CONFIG DATASET """
    if args.dataset == 'lung':
        val_dataset = DatasetLung('test', root_data=args.root_data, version=args.version) #!
    val_dataset.adjust_shape(multiple_of=32)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    """ INITIALIZE MODEL """
    model = init_model(args, img_size=val_dataset.inshape)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)

    """INITIALIZE LOSS FUNCTIONS """
    if args.similarity_loss == 'ncc':
        similarity_loss = NCC(args.device)
    if args.similarity_loss == 'nmi':
        similarity_loss = GlobalMutualInformationLoss()
    smooth_loss = Grad(penalty='l2', loss_mult=args.reg_weight)

    """ VALIDATION """
    metrics = validate_epoch(model, val_loader, run, args,
                   similarity_loss, smooth_loss)

    df = pd.DataFrame(metrics)
    csv_path = '{}/csv/{}_{}_ep-{:04d}.csv'.format(args.root_output, args.run_nr, args.network, epoch)
    df.to_csv(csv_path)
    df.to_pickle(csv_path.replace('csv', 'pkl'))
    run.stop()
