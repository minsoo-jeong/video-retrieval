from collections import OrderedDict
from torchvision import models
import torch
import torch.nn.functional as F
import einops

import math
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RMAC(torch.nn.Module):

    def __init__(self, L=[3]):
        super(RMAC, self).__init__()
        self.L = L

    def forward(self, x):
        return self.region_pooling(x, L=self.L)

    def region_pooling(self, x, L=[3]):
        ovr = 0.4  # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

        W = x.shape[3]
        H = x.shape[2]

        w = min(W, H)
        w2 = math.floor(w / 2.0 - 1)

        b = (max(H, W) - w) / (steps - 1)
        (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd, Hd = 0, 0
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        vecs = []
        for l in L:
            wl = math.floor(2 * w / (l + 1))
            wl2 = math.floor(wl / 2 - 1)

            if l + Wd == 1:
                b = 0
            else:
                b = (W - wl) / (l + Wd - 1)
            cenW = torch.floor(wl2 + torch.tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
            if l + Hd == 1:
                b = 0
            else:
                b = (H - wl) / (l + Hd - 1)
            cenH = torch.floor(wl2 + torch.tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

            for i in cenH.long().tolist():
                v = []
                for j in cenW.long().tolist():
                    if wl == 0:
                        continue
                    R = x[:, :, i: i + wl, j: j + wl]
                    vt = F.adaptive_max_pool2d(R, (1, 1))
                    v.append(vt)
                vecs.append(torch.cat(v, dim=3))
        vecs = torch.cat(vecs, dim=2)
        return vecs


class Resnet50(torch.nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.features = torch.nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
       [N, C, H ,W] = > [N, C]
       """
        x = self.features(x)
        x = self.pool(x)
        x = F.normalize(x, p=2, dim=1).squeeze(-1).squeeze(-1)

        return x


class Resnet50_RMAC(torch.nn.Module):
    def __init__(self, reduction=None):
        super(Resnet50_RMAC, self).__init__()
        self.features = torch.nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = RMAC()
        self.reduction = reduction if reduction in ('min', 'max', 'sum', 'mean', 'prod') else None

    def forward(self, x):
        """
       [N, C, H ,W] = > [N, C, R] if reduction is None, else [N, C]
       """
        x = self.features(x)
        x = self.pool(x)
        x = F.normalize(x, p=2, dim=1)

        if self.reduction is not None:
            x = einops.reduce(x, 'n c h w -> n c', self.reduction)
            x = F.normalize(x, p=2, dim=1)
        else:
            x = einops.rearrange(x, 'n c h w -> n c (h w)')

        return x


class SimpleMLP(torch.nn.Module):
    def __init__(self, dim, output_dim):
        super(SimpleMLP, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, output_dim),
        )
        self.bn = torch.nn.BatchNorm1d(dim)

    def forward(self, x, num_frames=None):
        # [N, T, C] => [N, C]

        x = einops.rearrange(x, 'n t c -> n c t')
        x = self.bn(x)

        x = self.pool(x).squeeze(-1)
        # x = F.normalize(x, p=2, dim=1)

        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)

        return x


class Transformer(torch.nn.Module):

    def __init__(self, dim=1024, nhead=8, nlayers=1, dropout=0.1):
        super(Transformer, self).__init__()

        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.bn = torch.nn.BatchNorm1d(dim)

        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=dim,
                                                          nhead=nhead,
                                                          dim_feedforward=2048,
                                                          dropout=dropout,
                                                          batch_first=True
                                                          )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.mlp = None

    def forward(self, x, num_frames=None):
        N, T, C = x.shape
        if num_frames is None:
            num_frames = torch.tensor([T for _ in range(N)])

        num_frames = num_frames.unsqueeze(-1).cuda()  # (N, 1)

        frame_mask = (0 < num_frames - torch.arange(0, T).cuda()).float()  # (N, T)

        x = einops.rearrange(x, 'n t c -> n c t')
        x = self.bn(x)
        x = einops.rearrange(x, 'n c t -> n t c')

        output = self.transformer_encoder(x, src_key_padding_mask=frame_mask)
        output = output * frame_mask.unsqueeze(-1)

        output = torch.sum(output, dim=1) / num_frames  # (N, C)
        embedding = F.normalize(output, p=2, dim=1)

        return embedding


class LSTM(torch.nn.Module):
    def __init__(self, dim=2048, hidden_dim=1024, n_layers=2, dropout=0.2):
        super(LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=dim,
                                  hidden_size=hidden_dim,
                                  num_layers=n_layers,
                                  dropout=dropout,
                                  batch_first=True)

        self.bn = torch.nn.BatchNorm1d(dim)

    def forward(self, x, num_frames=None):
        N, T, C = x.shape

        if num_frames is None:
            num_frames = torch.tensor([T for _ in range(N)])

        x = einops.rearrange(x, 'n t c -> n c t')
        x = self.bn(x)
        x = einops.rearrange(x, 'n c t -> n t c')

        x = pack_padded_sequence(x, num_frames.cpu(), batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()

        output, (h_n, c_n) = self.lstm(x, None)  # output x is the last hidden state output

        output, input_sizes = pad_packed_sequence(output, batch_first=True)

        x = torch.sum(output, dim=1, keepdim=False)  # [N, C]

        x = x / input_sizes.cuda().unsqueeze(-1)

        x = F.normalize(x, p=2, dim=1)

        return x


class LateTemporal(torch.nn.Module):
    def __init__(self, spatial_encoder, temporal_encoder):
        super(LateTemporal, self).__init__()
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder

    def forward(self, x):
        pass


class Resnet50_imac(torch.nn.Module):
    pass


class Resnet50_L3iRMAC(torch.nn.Module):
    pass


if __name__ == '__main__':
    from torchsummary import summary

    summary(Resnet50(), input_size=(3, 224, 224), device='cpu')
    # print(resnet.features.layer1)

    exit()

    # m = LSTM(dim=2048, hidden_dim=1024).cuda()
    m = Transformer(dim=2048).cuda()

    for i in range(100):
        x = torch.rand((3, 10, 2048)).cuda()
        num_frames = torch.tensor([i + 1 for i in range(x.shape[0])])
        # x = torch.rand((2, 3, 224, 224)).cuda()
        a = m(x, num_frames)
        print(a.shape)
        exit()
        # print(a)
