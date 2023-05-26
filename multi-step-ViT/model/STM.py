
import torch
import torch.nn as nn
import torch.nn.functional as F


def identity_grid(shape, unity=False):
    if unity:
        vectors = [torch.arange(0, s) / (s - 1) for s in shape]
    else:
        vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)
    return grid

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, shape, mode='bilinear', unity=False):
        super().__init__()
        self.shape = shape
        self.mode = mode
        self.unity = unity
        grid = identity_grid(shape=shape, unity=unity)
        self.register_buffer('grid', grid)
        # self.grid = identity_grid(shape=shape, unity=unity).to(device)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        if self.unity:
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] - 0.5)
        else:
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs.flip([-1])
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

    def warp_landmarks(self, src, flow):
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        flow = flow.permute(0, 2, 3, 4, 1)
        idx = torch.round(src).to(torch.int64)
        displacements = flow[:, idx[:, 0].tolist(), idx[:, 1].tolist(), idx[:, 2].tolist(), :].squeeze()
        if self.unity:
            displacements = displacements * torch.Tensor(data=flow.shape[1:4]).to(flow.device)
        points_warped = src + displacements
        return points_warped

class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, shape, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step
        grid = identity_grid(shape)
        self.register_buffer('grid', grid)

    def forward(self, velocity):
        flow = velocity / (2.0 ** self.time_step)
        for _ in range(self.time_step):
            grid = self.grid + flow.permute(0, 2, 3, 4, 1)
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class Re_SpatialTransformer(nn.Module):
    def __init__(self, shape, mode='bilinear', unity=False):
        super(Re_SpatialTransformer, self).__init__()
        self.stn_dvf = SpatialTransformer(shape=shape, mode='bilinear', unity=unity)
        self.stn = SpatialTransformer(shape=shape, mode=mode, unity=unity)

    def forward(self, src, flow):
        flow = -1 * self.stn_dvf(flow, flow)
        return self.stn(src, flow)
