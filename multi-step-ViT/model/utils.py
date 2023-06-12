import os

import torch

UNITY = False
def init_model(args, img_size):
    if args.network == 'transmorph':
        from model.TransMorph.configs import get_3DTransMorph_config
        from model.TransMorph import TransMorph as TransMorph
        config = get_3DTransMorph_config(img_size=tuple(img_size), args=args)
        model = TransMorph.TransMorph(config, unity=UNITY)
    elif args.network == 'msvit':
        from model.MultiStepViT.MultiStepViT import MultiStepViT
        from model.MultiStepViT.configs import get_MultiStepViT_config
        config = get_MultiStepViT_config(img_size=tuple(img_size),
                                             vit_steps=args.vit_steps,
                                             patch_size=args.patch_size,
                                             stages=args.stages,
                                             embed_dim=args.embed_dim,
                                             depths=args.depths,
                                             num_heads=args.num_heads,
                                             window_size=args.window_size,
                                             if_transskip=True,
                                             if_convskip=False,
                                             rpe=False)
        model = MultiStepViT(config, unity=UNITY)
    elif args.network == 'msvit_2':
        from model.MultiStepViT.MultiStepViT import MultiStepViT_2
        from model.MultiStepViT.configs import get_MultiStepViT_config
        config = get_MultiStepViT_config(img_size=tuple(img_size),
                                             vit_steps=args.vit_steps,
                                             patch_size=args.patch_size,
                                             stages=args.stages,
                                             embed_dim=args.embed_dim,
                                             depths=args.depths,
                                             num_heads=args.num_heads,
                                             window_size=args.window_size,
                                             if_transskip=True,
                                             if_convskip=False,
                                             rpe=False)
        model = MultiStepViT_2(config, unity=UNITY)
    return model

def save_model(model, args, epoch, run=None):
    if not os.path.exists('{}/{}'.format(args.root_checkpoints, args.run_nr)):
        os.mkdir('{}/{}'.format(args.root_checkpoints, args.run_nr))

    model_path = '{}/{}/{}_{}_ep-{:04d}.pt'.format(args.root_checkpoints, args.run_nr, args.run_nr, args.network, epoch)
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, model_path)
    run["model/path"] = model_path

class ScaledTanH(torch.nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def forward(self, input):
        return torch.tanh(input) * self.scaling

    def __repr__(self):
        return self.__class__.__name__ + "(" + "scaling = " + str(self.scaling) + ")"


class BiasedTanh(torch.nn.Module):
    def __init__(self, scale_in=1.0, scale_out=1.0, bias=0.0):
        super().__init__()
        self.scale_in = scale_in
        self.scale_out = scale_out
        self.bias = bias

    def forward(self, input):
        return torch.tanh(input * self.scale_in) * self.scale_out + self.bias

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_in="
            + str(self.scale_in)
            + ", "
            + "scale_out="
            + str(self.scale_out)
            + ", "
            + "bias="
            + str(self.bias)
            + ")"
        )


class ScalingAF(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor ** torch.tanh(input)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "scale_factor="
            + str(self.scale_factor)
            + ")"
        )


# Bspline functions
def bspline_kernel_nd(t, order, dtype=float):
    tpowers = t ** torch.arange(order, 0 - 1, -1, dtype=dtype) # print(tpowers)
    if order == 1:
        return tpowers @ torch.tensor(((-1, 1), (1, 0)), dtype=dtype)
    elif order == 2:
        return (
            tpowers
            @ torch.tensor(((1, -2, 1), (-2, 2, 0), (1, 1, 0)), dtype=dtype)
            / 2.0
        )
    elif order == 3:
        return (
            tpowers
            @ torch.tensor(
                ((-1, 3, -3, 1), (3, -6, 3, 0), (-3, 0, 3, 0), (1, 4, 1, 0)),
                dtype=dtype,
            )
            / 6.0
        )


def bspline_convolution_kernel(upsampling_factors, order, dtype=float):
    ndim = len(upsampling_factors)
    for i, us_factor in enumerate(upsampling_factors):
        t = torch.linspace(1 - (1 / us_factor), 0, us_factor)
        ker1D = bspline_kernel_nd(t[:, None], order, dtype).T.flatten() #print(ker1D)
        shape = (1,) * i + ker1D.shape + (1,) * (ndim - 1 - i)
        try:
            kernel = kernel * ker1D.view(shape)
        except NameError:
            kernel = ker1D.view(shape)
    return kernel