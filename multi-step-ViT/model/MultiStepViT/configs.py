'''
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import ml_collections

'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
rpe (bool): Enable relative position embedding. Default: True
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
'''


def get_VitBase_config(img_size,
                       patch_size,
                       stages=4,
                       embed_dim=96,
                       depths=2,
                       num_heads=4,
                       window_size=5,
                       if_transskip=True,
                       if_convskip=False,
                       rpe=False):
    config = ml_collections.ConfigDict()
    config.if_transskip = if_transskip
    config.if_convskip = if_convskip
    config.patch_size = patch_size
    config.in_chans = 2
    config.embed_dim = embed_dim
    config.window_size = tuple(3 * [window_size])
    if stages > 1:
        config.depths = tuple(stages * [depths])
        config.num_heads = tuple(stages * [num_heads])
    else:
        config.depths = (depths,)
        config.num_heads = (num_heads,)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = rpe
    config.patch_norm = True
    config.use_checkpoint = False
    if stages > 1:
        config.out_indices = tuple(range(stages))
    else:
        config.out_indices = (0,)
    config.img_size = img_size
    return config

def get_MultiStepViT_config(img_size,
                                vit_steps=2,
                                patch_size=[4, 4],
                                stages=[4, 4],
                                embed_dim=[96, 96],
                                depths=[2, 2],
                                num_heads=[4, 4],
                                window_size=[5, 5],
                                if_transskip=True,
                                if_convskip=False,
                                rpe=False):
    config = {'N': vit_steps}
    for n in range(vit_steps):
        config['config_' + str(n + 1)] = get_VitBase_config(img_size,
                                                            patch_size=patch_size[n],
                                                            stages=stages[n],
                                                            embed_dim=embed_dim[n],
                                                            depths=depths[n],
                                                            num_heads=num_heads[n],
                                                            window_size=window_size[n],
                                                            if_transskip=if_transskip,
                                                            if_convskip=if_convskip,
                                                            rpe=rpe)
    return config