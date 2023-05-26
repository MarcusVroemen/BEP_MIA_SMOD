from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from model.MultiStepViT.transformations import CubicBSplineFFDTransform
from model.STM import SpatialTransformer
from model.TransMorph.TransMorph import SwinTransformer, RegistrationHead, BasicLayerUp
from model.utils import bspline_convolution_kernel


class BsplineUpsampleBlock(nn.Module):
    """
    third order bspline. upsampling via transposed convolutions.
    """

    def __init__(self, shape, upsampling_factors: Tuple[int], order: int = 3):
        super().__init__()
        self.upsampling_factors = upsampling_factors
        bspline_kernel = self.make_bspline_kernel(self.upsampling_factors, order=order)
        kernel_size = bspline_kernel.shape
        crop_size = tuple(int(el * 3 / 8) if el != 5 else 2 for el in kernel_size)
        upsampler = nn.ConvTranspose3d(1, 1, kernel_size, stride=self.upsampling_factors, padding=crop_size, bias=False)
        upsampler.weight = nn.Parameter(bspline_kernel[None, None], requires_grad=False)
        self.upsampler = upsampler
        self.output_shape = shape

    @staticmethod
    def make_bspline_kernel(upsampling_factors, order, dtype=torch.float32):
        # TODO: solve the issues with kernel shapes
        bspline_kernel = bspline_convolution_kernel(upsampling_factors, order=order, dtype=dtype)
        if (np.array(bspline_kernel.shape[::-1]) == 4).any() or (
                np.array(bspline_kernel.shape[::-1]) == 2
        ).any():  # hack to deal with 1 strides and kernel size of 4
            padding = list()
            for s in bspline_kernel.shape[::-1]:
                if s == 4 or s == 2:
                    padding.extend([1, 0])
                else:
                    padding.extend([0, 0])
            bspline_kernel = F.pad(bspline_kernel, padding, mode="constant")
        return bspline_kernel

    def create_dvf(self, bspline_parameters, output_shape):
        shape = bspline_parameters.shape
        dvf = self.upsampler(
            bspline_parameters.view((shape[0] * 3, 1) + shape[2:]),
            output_size=output_shape,
        )
        newshape = dvf.shape
        return dvf.view((shape[0], 3) + newshape[2:])

    def forward(self, bspline_coefficients):
        dvf = self.create_dvf(bspline_coefficients, output_shape=self.output_shape)
        return dvf


class PatchExpand(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.up_conv = nn.ConvTranspose3d(dim, out_dim, (2, 2, 2), stride=2, padding=0)
        self.norm = norm_layer(out_dim)

    def forward(self, x, H, W, T):
        """
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        x = x.view(B, H, W, T, C)
        x = rearrange(x, 'b d h w c -> b c d h w')

        x = self.up_conv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = x.view(B, -1, self.out_dim)
        x = self.norm(x)
        return x


class SwinTransformerDecoder(nn.Module):
    r""" Swin Transformer --> rebuild to decoder """

    def __init__(self,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,

                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 out_indices=(3,),
                 frozen_stages=-1,
                 use_checkpoint=False, ):
        super().__init__()
        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.up_layers = nn.ModuleList()
        self.up_layers.append(PatchExpand(dim=int(embed_dim * 2 ** (self.num_layers - 1)),
                                          out_dim=int(embed_dim * 2 ** (self.num_layers - 2)),
                                          norm_layer=norm_layer))
        for i_layer in reversed(range(self.num_layers - 1)):
            up_layer = BasicLayerUp(
                dim=int(embed_dim * 2 ** (i_layer + 1)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand if i_layer != 0 else None,
                use_checkpoint=use_checkpoint)
            self.up_layers.append(up_layer)

        num_features = [int(embed_dim * 2 ** i) for i in reversed(range(self.num_layers))]
        self.num_features = num_features
        self.norm = norm_layer(2 * embed_dim)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, out_feats):
        """Forward function."""
        x = out_feats.pop()
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.up_layers[0](x, Wh, Ww, Wt)
        Wh, Ww, Wt = (Wh) * 2, (Ww) * 2, (Wt) * 2

        for i, skip in zip(range(1, self.num_layers), out_feats[::-1]):
            layer = self.up_layers[i]
            skip = skip.flatten(2).transpose(1, 2)
            x = torch.cat([x, skip], dim=2)
            x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)

        x = self.norm(x)
        out = x.view(-1, Wh, Ww, Wt, 2 * self.num_features[-1]).permute(0, 4, 1, 2, 3).contiguous()
        return out


class ViTBase(nn.Module):
    def __init__(self, config, unity=False):
        super(ViTBase, self).__init__()
        # initialize
        self.config = config
        self.unity = unity
        self.num_layers = len(config.depths)
        embed_dim = config.embed_dim
        patch_size = config.patch_size
        self.swin_transformer = SwinTransformer(patch_size=config.patch_size,
                                                in_chans=config.in_chans,
                                                embed_dim=config.embed_dim,
                                                depths=config.depths,
                                                num_heads=config.num_heads,
                                                window_size=config.window_size,
                                                mlp_ratio=config.mlp_ratio,
                                                qkv_bias=config.qkv_bias,
                                                drop_rate=config.drop_rate,
                                                drop_path_rate=config.drop_path_rate,
                                                ape=config.ape,
                                                spe=config.spe,
                                                rpe=config.rpe,
                                                patch_norm=config.patch_norm,
                                                use_checkpoint=config.use_checkpoint,
                                                out_indices=config.out_indices,
                                                pat_merg_rf=config.pat_merg_rf,
                                                )

        # stochastic depth
        self.swin_transformer_decoder = SwinTransformerDecoder(embed_dim=config.embed_dim,
                                                               depths=config.depths,
                                                               num_heads=config.num_heads,
                                                               window_size=config.window_size,
                                                               mlp_ratio=config.mlp_ratio,
                                                               qkv_bias=config.qkv_bias,
                                                               drop_rate=config.drop_rate,
                                                               drop_path_rate=config.drop_path_rate,
                                                               use_checkpoint=config.use_checkpoint,
                                                               out_indices=config.out_indices
                                                               )

        self.reg_head = RegistrationHead(
            in_channels=embed_dim * 2,
            out_channels=3,
            kernel_size=3,
        )

        self.upsample_bspline = BsplineUpsampleBlock(config.img_size,
                                                     upsampling_factors=(patch_size, patch_size, patch_size))
        self.stl = SpatialTransformer(config.img_size, unity=unity)
        self.stl_binary = SpatialTransformer(config.img_size, mode='nearest', unity=unity)

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, source, target):
        x = torch.cat((source, target), dim=1)
        out_feats = self.swin_transformer(x)
        out = self.swin_transformer_decoder(out_feats)

        bspline_params = self.reg_head(out)
        # st = time()
        dvf = self.upsample_bspline.forward(bspline_params)
        # et = time()
        # print('Execution time - upsample_bspline: ', et-st, 'seconds')

        source_warped = self.stl(source, dvf)
        return dvf, source_warped

class ViTBase_2(nn.Module):
    def __init__(self, config, unity=False):
        super(ViTBase_2, self).__init__()
        # initialize
        self.config = config
        self.unity = unity
        self.num_layers = len(config.depths)
        embed_dim = config.embed_dim
        patch_size = config.patch_size
        self.swin_transformer = SwinTransformer(patch_size=config.patch_size,
                                                in_chans=config.in_chans,
                                                embed_dim=config.embed_dim,
                                                depths=config.depths,
                                                num_heads=config.num_heads,
                                                window_size=config.window_size,
                                                mlp_ratio=config.mlp_ratio,
                                                qkv_bias=config.qkv_bias,
                                                drop_rate=config.drop_rate,
                                                drop_path_rate=config.drop_path_rate,
                                                ape=config.ape,
                                                spe=config.spe,
                                                rpe=config.rpe,
                                                patch_norm=config.patch_norm,
                                                use_checkpoint=config.use_checkpoint,
                                                out_indices=config.out_indices,
                                                pat_merg_rf=config.pat_merg_rf,
                                                )

        # stochastic depth
        self.swin_transformer_decoder = SwinTransformerDecoder(embed_dim=config.embed_dim,
                                                               depths=config.depths,
                                                               num_heads=config.num_heads,
                                                               window_size=config.window_size,
                                                               mlp_ratio=config.mlp_ratio,
                                                               qkv_bias=config.qkv_bias,
                                                               drop_rate=config.drop_rate,
                                                               drop_path_rate=config.drop_path_rate,
                                                               use_checkpoint=config.use_checkpoint,
                                                               out_indices=config.out_indices
                                                               )

        self.reg_head = RegistrationHead(
            in_channels=embed_dim * 2,
            out_channels=3,
            kernel_size=3,
        )

        self.upsample_bspline = CubicBSplineFFDTransform(img_size=config.img_size, ndim=3,
                                                         svf=False, cps=(patch_size+1, patch_size+1, patch_size+1))
        # self.upsample_bspline = BsplineUpsampleBlock(config.img_size,
                                                     # upsampling_factors=(patch_size, patch_size, patch_size))
        self.stl = SpatialTransformer(config.img_size, unity=unity)
        self.stl_binary = SpatialTransformer(config.img_size, mode='nearest', unity=unity)

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, source, target):
        x = torch.cat((source, target), dim=1)
        out_feats = self.swin_transformer(x)
        out = self.swin_transformer_decoder(out_feats)

        bspline_params = self.reg_head(out)
        # st = time()
        dvf = self.upsample_bspline(bspline_params)
        # et = time()
        # print('Execution time - upsample_bspline: ', et-st, 'seconds')

        source_warped = self.stl(source, dvf)
        return dvf, source_warped, target, source

class MultiStepViT(nn.Module):
    def __init__(self, config, unity=False):
        super(MultiStepViT, self).__init__()
        self.level = config['N']
        self.sub_networks = nn.ModuleList()
        for n in range(1, self.level + 1):
            self.sub_networks.append(ViTBase(config['config_' + str(n)], unity=unity))
        self.stl = self.sub_networks[-1].stl
        self.stl_binary = self.sub_networks[-1].stl_binary

    def set_level(self, level):
        self.level = level

    def composed_transform(self, dvf_list):
        # Use this transformed grid to interpolated DVF1 (first transform)
        for i in reversed(range(1, len(dvf_list))):
            # print(i)
            dvf_temp = dvf_list.pop(i)
            dvf_list[i - 1] = self.stl(src=dvf_list[i - 1], flow=dvf_temp) + dvf_temp
        dvf = dvf_list[0]
        return dvf

    def forward(self, source, target):
        source_temp = source #.detach().clone()
        dvfs_intermediate = list()
        for n, subnetwork in zip(range(1, self.level + 1), self.sub_networks):
            # print(n)
            dvf_intermediate, source_warped_intermediate = subnetwork(source_temp, target)
            dvfs_intermediate.append(dvf_intermediate)
            source_temp = source_warped_intermediate

        # get deformed moving image
        # print(dvfs_intermediate)
        dvf = self.composed_transform(dvfs_intermediate)
        source_warped = self.stl(source, dvf)
        return dvf, source_warped

class MultiStepViT_2(nn.Module):
    def __init__(self, config, unity=False):
        super(MultiStepViT_2, self).__init__()
        self.level = config['N']
        self.sub_networks = nn.ModuleList()
        for n in range(1, self.level + 1):
            self.sub_networks.append(ViTBase_2(config['config_' + str(n)], unity=unity))
        self.stl = self.sub_networks[-1].stl
        self.stl_binary = self.sub_networks[-1].stl_binary

    def set_level(self, level):
        self.level = level

    def composed_transform(self, dvf_list):
        # Use this transformed grid to interpolated DVF1 (first transform)
        for i in reversed(range(1, len(dvf_list))):
            # print(i)
            dvf_temp = dvf_list.pop(i)
            dvf_list[i - 1] = self.stl(src=dvf_list[i - 1], flow=dvf_temp) + dvf_temp
        dvf = dvf_list[0]
        return dvf

    def forward(self, source, target):
        source_warped_intermediate = source
        dvfs_intermediate = list()
        for n, subnetwork in zip(range(1, self.level + 1), self.sub_networks):
            dvf_intermediate, source_warped_intermediate, \
            target_intermediate, source_intermediate = subnetwork(source_warped_intermediate, target)
            dvfs_intermediate.append(dvf_intermediate)

        # get deformed moving image
        dvf = self.composed_transform(dvfs_intermediate)
        source_warped = self.stl(source, dvf)
        return dvf, source_warped