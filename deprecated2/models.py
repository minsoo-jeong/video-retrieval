from collections import OrderedDict
from typing import Union, List

import einops
import pytorchvideo
import torch
import torchvision
import torchvision.models
from torchsummary import summary

from futures.layers import *
from futures.pca import PCA


class Encoder(nn.Module):
    def __init__(self, clip_encoder, video_encoder):
        super(Encoder, self).__init__()
        self.clip_encoder = clip_encoder
        self.video_encoder = video_encoder

    def forward(self, x, length, return_clip_emb=False):
        b, n, c, t, h, w = x.shape
        x = einops.rearrange(x, 'b n c t h w -> (b n) c t h w' if t != 1 else 'b n c t h w -> (b n) (c t) h w', t=t)
        mask = torch.arange(n).cuda().unsqueeze(0).lt(length.unsqueeze(-1)).unsqueeze(-1)  # [batch, clip, 1]

        clip_emb = self.forward_clip_encoder(x, 64)
        clip_emb = einops.rearrange(clip_emb, '(b n) c -> b n c', b=b, n=n)
        clip_emb = clip_emb * mask

        embeddings = self.video_encoder(clip_emb, length)
        (video_emb, clip_emb) = embeddings if len(embeddings) > 1 else (embeddings, clip_emb)

        if return_clip_emb:
            return video_emb, clip_emb
        return video_emb

    def forward_clip_encoder(self, x, split=None):
        with torch.no_grad():
            if isinstance(split, int):
                _emb = []
                for _x in torch.split(x, split):
                    _emb.append(self.clip_encoder(_x))
                clip_emb = torch.cat(_emb)
            else:
                clip_emb = self.clip_encoder(x)
        return clip_emb


##############
# Clip encoder
##############

class R50(object):
    checkpoints = {
        'moco_v1': '/workspace/futures/pretrained/moco_v1/moco_v1_200ep_pretrain.pth.tar',
        'moco_v2': '/workspace/futures/pretrained/moco_v2/moco_v2_800ep_pretrain.pth.tar',
        'moco_v2_ep200': '/workspace/futures/pretrained/moco_v2/moco_v2_200ep_pretrain.pth.tar',
        'moco_v2_ep800': '/workspace/futures/pretrained/moco_v2/moco_v2_800ep_pretrain.pth.tar',
        'moco_v3': '/workspace/futures/pretrained/moco_v3/r-50-1000ep.pth.tar',
        'moco_v3_ep1000': '/workspace/futures/pretrained/moco_v3/r-50-1000ep.pth.tar',
        'dino': '/workspace/futures/pretrained/dino/dino_resnet50_pretrain.pth',
        'byol': '/workspace/futures/pretrained/byol/pretrain_res50x1.pth.tar'
    }

    def __init__(self, pretrained: Union[bool, str] = True, exclude: List[str] = None):
        super(R50, self).__init__()
        self._resnet50 = torchvision.models.resnet50(pretrained=True)
        if isinstance(pretrained, str):
            self.load_pretrained_checkpoints(pretrained.lower())

        exclude = exclude if exclude else []
        for layer in exclude:
            if getattr(self._resnet50, layer, None):
                delattr(self._resnet50, layer)

    def load_pretrained_checkpoints(self, pretrained: str):
        if pretrained not in self.checkpoints.keys():
            raise NotImplementedError(f'Unsupported pretrained checkpoint ({pretrained}).')

        ckpt = torch.load(self.checkpoints[pretrained])
        if pretrained.startswith('moco_v1') or pretrained.startswith('moco_v2'):
            state = {k.replace('module.encoder_q.', ''): v for k, v in ckpt['state_dict'].items() if
                     not k.startswith('module.encoder_q.fc')}
        elif pretrained.startswith('moco_v3'):
            state = {k.replace('module.base_encoder.', ''): v for k, v in ckpt['state_dict'].items()}
        else:
            state = ckpt
        self._resnet50.load_state_dict(state, strict=False)

    @property
    def resnet50(self):
        return self._resnet50


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = R50(pretrained='abc').resnet50

        # self.layers = nn.Sequential(OrderedDict(list(tv_models.resnet50(pretrained=True).named_children())[:-1]))
        self.layers = nn.Sequential(OrderedDict(list(resnet.named_children())[:-1]))

    def forward(self, x):
        # [N, C, H ,W] = > [N, C]
        x = self.layers(x).squeeze(-1).squeeze(-1)

        x = F.normalize(x, p=2, dim=1)

        return x


import vits


class Vit(nn.Module):
    def __init__(self, arch, **kwargs):
        super(Vit, self).__init__()
        self.vit = vits.__dict__[arch]()
        print(self.vit)


class Resnet50_GAP(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50_GAP, self).__init__()
        resnet = R50(pretrained=pretrained).resnet50
        self.layers = nn.Sequential(OrderedDict(list(resnet.named_children())[:-1]))  # Last: Global Average Pooling

    def forward(self, x):
        # [N, C, H ,W] = > [N, C]
        x = self.layers(x).squeeze(-1).squeeze(-1)

        x = F.normalize(x, p=2, dim=1)

        return x


class Resnet50_GeM(nn.Module):
    def __init__(self, pretrained: Union[bool, str] = True):
        super(Resnet50_GeM, self).__init__()
        resnet = R50(pretrained=pretrained).resnet50
        self.layers = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2]))
        self.gem = GeM(p=3)

    def forward(self, x):
        # [N, C, H ,W] = > [N, C]
        x = self.layers(x)
        x = self.gem(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, p=2, dim=1)
        return x


class Resnet50_RMAC(nn.Module):
    def __init__(self, pretrained: Union[bool, str] = True):
        super(Resnet50_RMAC, self).__init__()
        resnet = R50(pretrained=pretrained).resnet50
        self.layers = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2]))
        self.rmac = RMAC()

    def forward(self, x):
        # [N, C, H ,W] = > [N, C]
        x = self.layers(x)
        x = self.rmac(x).squeeze(-1).squeeze(-1)
        x = F.normalize(x, p=2, dim=1)
        return x


class Resnet50_backbone(nn.Module):
    def __init__(self, pretrained: [bool, str] = True, exclude=None):
        super(Resnet50_backbone, self).__init__()
        resnet50 = R50(pretrained=pretrained, exclude=exclude).resnet50
        for k, v in resnet50.named_children():
            setattr(self, k, v)


class Resnet50_IRMAC(Resnet50_backbone):
    def __init__(self, pretrained: [bool, str] = True):
        super(Resnet50_IRMAC, self).__init__(pretrained=pretrained, exclude=['fc', 'avgpool'])

        self.concat = Concatenate()
        self.rmac = RMAC()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        l1 = self.rmac(x)

        x = self.layer2(x)
        l2 = self.rmac(x)

        x = self.layer3(x)
        l3 = self.rmac(x)

        x = self.layer4(x)
        l4 = self.rmac(x)

        x = self.concat(l1, l2, l3, l4, dim=1).squeeze(-1).squeeze(-1)

        return x


class Resnet50_IRMAC_TCA(Resnet50_IRMAC):
    def __init__(self, normalize=True, pretrained: [bool, str] = True):
        super(Resnet50_IRMAC_TCA, self).__init__(pretrained)
        self.rmac = RMAC_TCA(normalize=normalize)


class Resnet50_IRMAC_PCA(Resnet50_IRMAC):
    def __init__(self, pca_param, n_components=1024, whitening=True, pretrained: [bool, str] = True):
        super(Resnet50_IRMAC_PCA, self).__init__(pretrained)
        self.pca = PCA(n_components=n_components, whitening=whitening, parameters_path=pca_param)
        self.pca.load()

    def forward(self, x):
        x = super(Resnet50_IRMAC_PCA, self).forward(x)
        x = self.pca.infer(x)
        x = F.normalize(x)
        return x


class Resnet50_IRMAC_TCA_PCA(Resnet50_IRMAC):
    def __init__(self, pca_param, n_components=1024, whitening=True, normalize=True, pretrained: [bool, str] = True):
        super(Resnet50_IRMAC_TCA_PCA, self).__init__(pretrained)
        self.rmac = RMAC_TCA(normalize=normalize)
        self.pca = PCA(n_components=n_components, whitening=whitening, parameters_path=pca_param)
        self.pca.load()

    def forward(self, x):
        x = super(Resnet50_IRMAC_TCA_PCA, self).forward(x)
        x = self.pca.infer(x)
        x = F.normalize(x)
        return x


class TorchHubWrap(nn.Module):
    def __init__(self, model_name):
        super(TorchHubWrap, self).__init__()
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

        head = model.blocks[-1]
        if isinstance(head, pytorchvideo.models.head.ResNetBasicHead):
            projection_layer = head.pool
            model.blocks[-1] = projection_layer
        else:
            raise NotImplementedError('Unsupported model')

        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(-1).squeeze(-1).squeeze(-1)


class Transformer(torch.nn.Module):
    def __init__(self, dim=1024, nhead=8, nlayers=1, dropout=0.1):
        super(Transformer, self).__init__()

        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=dim,
                                                          nhead=nhead,
                                                          dim_feedforward=2048,
                                                          dropout=dropout,
                                                          batch_first=True
                                                          )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, x, length):
        n, t, c = x.shape
        length = length.unsqueeze(-1)  # [batch, 1]

        # [1, t] * [n, 1] => [n, t]
        mask = torch.arange(t).cuda().unsqueeze(0).lt(length)  # [batch, clip]

        outputs = self.transformer_encoder(x, src_key_padding_mask=mask.float())
        outputs = outputs * mask.unsqueeze(-1)  # (N, T, C)

        embedding = torch.sum(outputs, dim=1) / length  # (N, C)
        embedding = F.normalize(embedding, p=2, dim=1)

        outputs = F.normalize(outputs, p=2, dim=2)
        return embedding, outputs

    def forward_tca(self, x, num_frames=None):
        N, T, C = x.shape
        if num_frames is None:
            num_frames = torch.tensor([T for _ in range(N)])

        num_frames = num_frames.unsqueeze(-1).cuda()  # (N, 1)

        frame_mask = (0 < num_frames - torch.arange(0, T).cuda()).float()  # (N, T)

        x = einops.rearrange(x, 'n t c -> n c t')
        x = self.bn(x)
        x = einops.rearrange(x, 'n c t -> n t c')

        outputs = self.transformer_encoder(x, src_key_padding_mask=frame_mask)
        outputs = outputs * frame_mask.unsqueeze(-1)  # (N, T, C)

        embedding = torch.sum(outputs, dim=1) / num_frames  # (N, C)
        embedding = F.normalize(embedding, p=2, dim=1)
        outputs = F.normalize(outputs, p=2, dim=2)

        return embedding, outputs


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
        outputs = x

        embedding = einops.rearrange(outputs, 'n t c -> n c t')
        embedding = self.bn(embedding)
        embedding = self.pool(embedding).squeeze(-1)

        embedding = self.mlp(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, outputs


class Basic(torch.nn.Module):
    def __init__(self):
        super(Basic, self).__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, length):
        n, t, c = x.shape
        length = length.unsqueeze(-1)  # [batch, 1]
        mask = torch.arange(t).cuda().unsqueeze(0).lt(length).unsqueeze(-1)  # [batch, clip,1]

        outputs = x
        embedding = x * mask
        embedding = torch.sum(embedding, dim=1) / length  # [batch, dim]
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, outputs


if __name__ == '__main__':
    m = Vit('vit_base')

    exit()
    m = Resnet50_GeM()
    m = Resnet50_GAP(pretrained='byol')
    # summary(m, (3, 224, 224), device='cpu')
    a = torch.rand((2, 5, 3, 1, 224, 224))
    e = Encoder(m, Basic()).cuda()
    x = e(a.cuda(), torch.tensor([5, 1]).cuda(), return_clip_emb=False)
    print(x.shape)
    print(x[0].shape)
    print(x[0].shape)
    print(x[1].shape)

    exit()
    # print(m.state_dict().keys())

    m = Transformer(dim=8, nhead=8, nlayers=1, dropout=0).cuda()
    a = torch.rand((4, 5, 8)).cuda()
    b = torch.Tensor([1, 2, 3, 4]).cuda()
    m.eval()
    with torch.no_grad():
        c = m(a, b)
        d = m.forward_tca(a, b)
        print(c[0], d[0])
        print(c[1], d[1])

    # tv_models.video.r3d_18(pretrained=True)
    # model = torch.hub.load('facebookresearch/pytorchvideo', 'r2plus1d_r50', pretrained=True)

    # summary(model, (3, 13, 160, 160), device='cpu')

    # model = TorchHubWrap('x3d_m')
    # summary(model, (3, 16, 224, 224), device='cpu')
