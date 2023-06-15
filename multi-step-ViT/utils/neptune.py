"""
Neptune functions
"""
from argparse import Namespace


import neptune

import numpy as np

def string_2_list(string_list):
    return [int(val) for val in string_list.strip('][').split(', ')]

def log_dict_2_neptune(run, dict, prefix):
    for key, value in dict.items():
        try:
            run["{}/{}".format(prefix, key)].log(np.nanmean(value))
        except:
            pass

def log_descriptives_2_neptune(run, var, values):
    run["eval_{}/{}_mean".format(args.mode, col)] = metrics[col].mean()
    run[var + "_mean"] = values.mean()
    run[var + "_median"] = values.median()
    run[var + "_std"] = values.std()
    run[var + "_q10"] = values.quantile(q=0.1)
    run[var + "_q90"] = values.quantile(q=0.9)

def init_neptune(args):
    run = neptune.init_run(project='marcusvroemen/ViT-{}'.format(args.dataset),
                           source_files=['BEP_MIA_DIR/multi-step-ViT/*.py', 'BEP_MIA_DIR/multi-step-ViT/utils/*.py', 'BEP_MIA_DIR/multi-step-ViT/model/***', 'BEP_MIA_DIR/multi-step-ViT/datasets/*.py', 'BEP_MIA_DIR/multi-step-ViT/evaluation/*.py', 'BEP_MIA_DIR/multi-step-ViT/executor/*.py'],
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMjhiYTdiYS02ZDk1LTQ3ZWEtYThjMC03MTAzNzg4MGRkZTQifQ==',
                           mode=args.mode_neptune)
    run["parameters"] = vars(args)
    args.run_nr = run['sys/id'].fetch()
    epoch = 0
    return run, args, epoch

def re_init_neptune(args):
    run = neptune.init_run(project='marcusvroemen/ViT-{}'.format(args.dataset),
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMjhiYTdiYS02ZDk1LTQ3ZWEtYThjMC03MTAzNzg4MGRkZTQifQ==',
                           with_id=f'VIT-{args.run_nr}')
    args.run_nr = run['sys/id'].fetch()
    args.N = run['dataset/size'].fetch() 

    args_ = run['parameters'].fetch()
    args = dict(args_, **vars(args))
    args = Namespace(**args)
    for k in ['depths', 'embed_dim', 'num_heads', 'stages', 'window_size', 'patch_size']:
        try:
            args.__dict__[k] = string_2_list(args.__dict__[k])
        except:
            pass

    print(args)
    args.model_path = run['model/path'].fetch() 
    epoch = int(args.model_path[-6:-3])
    return run, args, epoch