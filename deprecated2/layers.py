import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return self.rmac(x, L=self.L, eps=self.eps)

    def rmac(self, x, L=3, eps=1e-6):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(1, L + 1):
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                    R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                    v += vt

        v = F.normalize(v).squeeze(-1).squeeze(-1)
        return v


class GeM(nn.Module):
    def __init__(self, p, output_size=1, eps=1e-6):
        super(GeM, self).__init__()
        assert p > 0
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)


class RMAC_TCA(nn.Module):
    def __init__(self, normalize=True):
        super(RMAC_TCA, self).__init__()
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x)
        p = int(min(x.size()[-2], x.size()[-1]) * 2 / 7)  # 28->8 14->4 7->2
        x = F.max_pool2d(x, kernel_size=(int(p + p / 2), int(p + p / 2)), stride=p)  # (n, c, 3, 3)
        x = x.view(x.size()[0], x.size()[1], -1)  # (n, c, 9)
        x = F.normalize(x)
        x = torch.sum(x, dim=-1)
        x = F.normalize(x)
        return x


class Concatenate(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Concatenate, self).__init__()
        # self.identity = nn.Identity(*args, **kwargs)

    def forward(self, *x, dim=1):
        x = torch.cat(x, dim=dim)
        # x = self.identity(x)
        return x
