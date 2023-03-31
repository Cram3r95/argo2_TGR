import math
import numpy as np
from fractions import gcd
from numbers import Number

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states
    
class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states
    
class LinearRes_sigmod(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes_sigmod, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.m = nn.Sigmoid()
        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.m(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.m(out)
        return out


# Conv layer with norm (gn or bn) and relu. 
class Conv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True, groups=1):
        super(Conv, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False, groups=groups)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act    

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True, groups=1):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False, groups=groups)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


# Post residual layer
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1, norm='GN', ng=32, act=True, groups=1):
        super(PostRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False, groups=groups)
        self.relu = nn.ReLU(inplace = True)
        
        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm2d(n_out)
            self.bn2 = nn.BatchNorm2d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.BatchNorm2d(n_out))
            else:
                exit('SyncBN has not been added!')    
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out

class no_pad_Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True, groups=1):
        super(no_pad_Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=False, groups=groups)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False, groups=groups)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x[:,:,2:]
        if self.act:
            out = self.relu(out)
        return out
    
class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True, groups=1):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False, groups=groups)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False, groups=groups),
                        nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()
        
    def forward(self, x):
        return x


def linear_interp(x, n_max):
    """Given a Tensor of normed positions, returns linear interplotion weights and indices.
    Example: For position 1.2, its neighboring pixels have indices 0 and 1, corresponding
    to coordinates 0.5 and 1.5 (center of the pixel), and linear weights are 0.3 and 0.7.
    Args:
        x: Normalizzed positions, ranges from 0 to 1, float Tensor.
        n_max: Size of the dimension (pixels), multiply x to get absolution positions.
    Returns: Weights and indices of left side and right side.
    """
    x = x * n_max - 0.5

    mask = x < 0
    x[mask] = 0
    mask = x > n_max - 1
    x[mask] = n_max - 1
    n = torch.floor(x)

    rw = x - n
    lw = 1.0 - rw
    li = n.long()
    ri = li + 1
    mask = ri > n_max - 1
    ri[mask] = n_max - 1

    return lw, li, rw, ri


def get_pixel_feat(fm, bboxes, pts_range):
    x, y = bboxes[:, 0], bboxes[:, 1]
    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    _, fm_h, fm_w = fm.size()
    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat = \
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1) +\
        (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1) +\
        (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1) +\
        (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    return feat


def get_roi_feat(fm, bboxes, roi_size, pts_range):
    """Given a set of BEV bboxes get their BEV ROI features.
    Args:
        fm: Feature map, float tensor, chw
        bboxes: BEV bboxes, n x 5 float tensor (cx, cy, wid, hgt, theta)
        roi_size: ROI size (number of bins), [int] or int
        pts_range: Range of points, tuple of ints, (x_min, x_max, y_min, y_max, z_min, z_max)
    Returns: Extracted features of size (num_roi, c, roi_size, roi_size).
    """
    if isinstance(roi_size, Number):
        roi_size = [roi_size, roi_size]

    cx, cy, wid, hgt, theta = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4]
    st = torch.sin(theta)
    ct = torch.cos(theta)
    num_bboxes = len(bboxes)

    rot_mat = bboxes.new().resize_(num_bboxes, 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct

    offset = bboxes.new().resize_(len(bboxes), roi_size[0], roi_size[1], 2)
    x_bin = (torch.arange(roi_size[1]).float().to(bboxes.device) + 0.5) / roi_size[1] - 0.5
    offset[:, :, :, 0] = x_bin.view(1, 1, -1) * wid.view(-1, 1, 1)
    y_bin = (torch.arange(roi_size[0] - 1, -1, -1).float().to(bboxes.device) + 0.5) / roi_size[0] - 0.5
    offset[:, :, :, 1] = y_bin.view(1, -1, 1) * hgt.view(-1, 1, 1)

    rot_mat = rot_mat.view(num_bboxes, 1, 1, 2, 2)
    offset = offset.view(num_bboxes, roi_size[0], roi_size[1], 2, 1)
    offset = torch.matmul(rot_mat, offset).view(num_bboxes, roi_size[0], roi_size[1], 2)

    x = cx.view(-1, 1, 1) + offset[:, :, :, 0]
    y = cy.view(-1, 1, 1) + offset[:, :, :, 1]
    x = x.view(-1)
    y = y.view(-1)

    x_min, x_max, y_min, y_max = pts_range[:4]
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    fm_c, fm_h, fm_w = fm.size()
    feat = fm.new().float().resize_(num_bboxes * roi_size[0] * roi_size[1], fm_c)
    mask = (x > 0) * (x < 1) * (y > 0) * (y < 1)
    x = x[mask]
    y = y[mask]

    xlw, xli, xhw, xhi = linear_interp(x, fm_w)
    ylw, yli, yhw, yhi = linear_interp(y, fm_h)
    feat[mask] = \
        (xlw * ylw).unsqueeze(1) * fm[:, yli, xli].transpose(0, 1) +\
        (xlw * yhw).unsqueeze(1) * fm[:, yhi, xli].transpose(0, 1) +\
        (xhw * ylw).unsqueeze(1) * fm[:, yli, xhi].transpose(0, 1) +\
        (xhw * yhw).unsqueeze(1) * fm[:, yhi, xhi].transpose(0, 1)
    feat[torch.logical_not(mask)] = 0
    feat = feat.view(num_bboxes, roi_size[0] * roi_size[1], fm_c)
    feat = feat.transpose(1, 2).contiguous().view(num_bboxes, -1, roi_size[0], roi_size[1])
    return feat

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
