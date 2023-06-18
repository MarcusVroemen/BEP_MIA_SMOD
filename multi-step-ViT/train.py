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

# import augmentation.GRYDS as AG
# import augmentation.SMOD as AS
import augmentations as AUG

torch.backends.cudnn.benchmark = True  # speed ups

base_path = "/home/bme001/20203531/BEP/BEP_MIA_DIR/BEP_MIA_DIR/"
# base_path = "C:/Users/Quinten Vroemen/Documents/MV_codespace/BEP_MIA_DIR/"
# base_path = "C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/"


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

# MultiStepViT architecture
parser.add_argument('--vit_steps', type=int, metavar='', default=2, help='the number of steps of the MultiStepViT')
parser.add_argument('--patch_size', type=int, metavar='', default=[8, 4], nargs='+', help='patch size')
parser.add_argument('--stages', type=int, metavar='', default=[3, 4], nargs='+', help='nr. of stages')
parser.add_argument('--embed_dim', type=int, metavar='', default=[48, 96], nargs='+', help='embedded dimensions') 
parser.add_argument('--depths', type=int, metavar='', default=[2,2], nargs='+', help='nr. of MSA blocks')   
parser.add_argument('--num_heads', type=int, metavar='', default=[4,4], nargs='+',                          
                    help='nr. of attention heads in the MSA block')
parser.add_argument('--window_size', type=int, metavar='', default=[2,2], nargs='+', help='window size')    

# Hyper-parameters for training
parser.add_argument('-loss', '--similarity_loss', type=str, metavar='', default='ncc',
                    help='similarity loss | nmi | ncc ')
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=1e-4, help='learning rate')
parser.add_argument("-rw", '--reg_weight', type=float, metavar='', default=0.5, help='regularization (smoothing) weight')
parser.add_argument('-ep', '--epochs', type=int, metavar='', default=5, help='nr of epochs you want to train on')
parser.add_argument('-bs', '--batch_size', type=int, metavar='', default=1,
                    help='batch size you want to use during training')

# Dataset
parser.add_argument('-set', '--dataset', type=str, metavar='', default='lung', help='dataset')
parser.add_argument('-v', '--version', type=str, metavar='', default='', help='preprocessing version')
parser.add_argument('--overfit', action='store_true', help='overfit on 1 image during training')
# Data augmentation 
parser.add_argument('-aug', '--augmentation', type=str, metavar='', default='none') 
parser.add_argument('-var', '--varyingaug', type=str, metavar='', default="False") 
parser.add_argument('-sig1', '--sigma1', type=int, metavar='', default=15000) 
parser.add_argument('-sig2', '--sigma2', type=int, metavar='', default=1500) 

args = parser.parse_args()
print(vars(args))
set_seed(args.random_seed)

if __name__ == "__main__":
    """ CONFIG NEPTUNE """
    args.mode = 'train'
    run, args, epoch = init_neptune(args)

    """ CONFIG DATASET """
    if args.dataset == 'lung':
        if args.varyingaug == "False":
            if args.augmentation == 'none':
                # args.augmentation = none
                train_dataset = DatasetLung('train', root_data=args.root_data, version=args.version)
                print("Init regular training data with size: ", len(train_dataset))
            else:
                # args.augmentation = gryds, gryds+real, SMOD, SMOD+real
                train_dataset = DatasetLung('train', root_data=args.root_data, version=args.version, folder_augment=args.augmentation)
                print(f"Init fixed training data {args.augmentation} with size: {len(train_dataset)}")
        
        elif args.varyingaug == "True":
            if 'SMOD' in args.augmentation:
                # args.augmentation = SMOD or SMOD+real
                dataset_original = DatasetLung('train', root_data=args.root_data, version=args.version)
                augmenter_SMOD = AUG.Augmentation_SMOD(root_data=args.root_data, original_dataset=dataset_original,
                                                sigma1=args.sigma1, sigma2=args.sigma2, plot=False, load_atlas=True, 
                                                load_preprocessing=True, save_preprocessing=False)
                train_dataset = AUG.DatasetLung(train_val_test='train', version='', root_data=args.root_data, 
                                                augmenter=augmenter_SMOD, augment=args.augmentation, save_augmented=False, phases='in_ex')
                print(f"Init SMOD training data {args.augmentation} with size: {len(train_dataset)}")
                
            elif 'gryds' in args.augmentation:
                # args.augmentation = gryds or gryds+real
                augmenter_gryds = AUG.Augmentation_gryds(args)
                train_dataset = AUG.DatasetLung(train_val_test='train', version='', root_data=args.root_data, 
                                            augmenter=augmenter_gryds, augment=args.augmentation, save_augmented=True, phases='in_ex')
                print(f"Init gryds* training data {args.augmentation} with size: {len(train_dataset)}")
            
            
            
    # val_dataset = DatasetLung('val', root_data=args.root_data, version=args.version)
    
    # train_dataset.adjust_shape(multiple_of=32)
    # val_dataset.adjust_shape(multiple_of=32)
    # if args.overfit:
    #     train_dataset.overfit_one(i=0)
    #     val_dataset = train_dataset
    #     val_dataset.train_val_test = 'val'
    # run["dataset/size"] = len(train_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # """ INITIALIZE MODEL """
    # model = init_model(args, img_size=train_dataset.inshape)
    # run["model/trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # run["model/architecture"] = model
    # model = model.to(args.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # if args.mode_neptune != 'debug':
    #     save_model(model, args, epoch, run)

    # """INITIALIZE LOSS FUNCTIONS """
    # if args.similarity_loss == 'ncc':
    #     similarity_loss = NCC(args.device)
    # if args.similarity_loss == 'nmi':
    #     similarity_loss = GlobalMutualInformationLoss()
    # smooth_loss = Grad(penalty='l2', loss_mult=args.reg_weight)

    # """ TRAINING """
    # print('\n----- Training -----')
    # # Baseline losses and metrics before training
    # validate_epoch(model, train_loader, run, args,
    #                similarity_loss, smooth_loss)
    # validate_epoch(model, val_loader, run, args,
    #                similarity_loss, smooth_loss)

    # # Train the model for the specified amount of epochs
    # epoch += 1
    # while epoch < args.epochs + 1:
    #     print(f'\n[epoch {epoch} / {args.epochs}]')
    #     # Train and validate for one epoch
    #     train_epoch(model, train_loader, optimizer, run, args,
    #                 similarity_loss, smooth_loss)
    #     metrics = validate_epoch(model, val_loader, run, args,
    #                    similarity_loss, smooth_loss)
    #     print(metrics)
    #     # Save the model each epoch
    #     epoch += 1
    #     if args.mode_neptune != 'debug' and epoch % 5 == 0:
    #         save_model(model, args, epoch, run)

    # df = pd.DataFrame(metrics)
    # print(df)
    # csv_path = '{}/csv/{}_{}_ep-{:04d}.csv'.format(args.root_output, args.run_nr, args.network, epoch - 1)
    # df.to_csv(csv_path)
    # df.to_pickle(csv_path.replace('csv', 'pkl'))
    # run.stop()
