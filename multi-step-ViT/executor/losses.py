import math
import numpy as np
import torch
import torch.nn.functional as F


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device, win=None):
        self.win = win
        self.device = device

    def forward(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

            # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class Grad:
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    @staticmethod
    def dvf_diff(input, dim):
        if dim == 0:
            diff = input[1:, :, :, :, :] - input[:-1, :, :, :, :]
        elif dim == 1:
            diff = input[:, 1:, :, :, :] - input[:, :-1, :, :, :]
        elif dim == 2:
            diff = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
        elif dim == 3:
            diff = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
        elif dim == 4:
            diff = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]
        return diff

    def forward(self, dvf):
        dz = torch.abs(self.dvf_diff(dvf, dim=2))
        dy = torch.abs(self.dvf_diff(dvf, dim=3))
        dx = torch.abs(self.dvf_diff(dvf, dim=4))

        if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad