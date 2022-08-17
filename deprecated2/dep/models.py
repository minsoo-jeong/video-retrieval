import torch.nn as nn
import torch

import einops
from typing import Union,List

from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, clip_encoder, video_encoder):
        super(Encoder, self).__init__()
        self.clip_encoder = clip_encoder

        self.video_encoder = video_encoder

    def forward(self, x, length, return_clip_emb=False):
        clip_emb = self.clip_encoder(x, length)

        video_emb = self.video_encoder(clip_emb, length)
        if len(video_emb) > 1:
            video_emb, clip_emb = video_emb

        embeddings = video_emb
        if return_clip_emb:
            embeddings = [embeddings, clip_emb]
        return embeddings


class Encoder2(nn.Module):
    def __init__(self, clip_encoder, video_encoder):
        super(Encoder2, self).__init__()
        self.clip_encoder = clip_encoder
        self.video_encoder = video_encoder

    def forward(self, x, length, return_clip_emb=False):
        b, n, c, t, h, w = x.shape
        x = einops.rearrange(x, 'b n c t h w -> (b n) c t h w' if t != 1 else 'b n c t h w -> (b n) (c t) h w', t=t)
        mask = torch.arange(n).cuda().unsqueeze(0).lt(length.unsqueeze(-1)).unsqueeze(-1)  # [batch, clip, 1]

        clip_emb = self.clip_encoder(x)
        clip_emb = einops.rearrange(clip_emb, '(b n) c -> b n c', b=b, n=n)
        clip_emb = clip_emb * mask

        video_emb = self.video_encoder(clip_emb, length)
        if len(video_emb) > 1:
            video_emb, clip_emb = video_emb
            clip_emb = clip_emb * mask

        embeddings = video_emb
        if return_clip_emb:
            embeddings = [embeddings, clip_emb]
        return embeddings


class ClipEncoder(nn.Module):
    def __init__(self, encoder):
        super(ClipEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, x, length):
        # [batch, n_clip, channel, clip_len, height, width]
        b, n, c, t, h, w = x.shape
        x = einops.rearrange(x, 'b n c t h w -> (b n) c t h w' if t != 1 else 'b n c t h w -> (b n) (c t) h w', t=t)
        mask = torch.arange(n).cuda().unsqueeze(0).lt(length.unsqueeze(-1)).unsqueeze(-1)  # [batch, clip, 1]

        x = self.encoder(x)
        x = einops.rearrange(x, '(b n) c -> b n c', b=b, n=n)  # [batch, n clip, 2048]
        x = x * mask

        return x


class VideoEncoder(nn.Module):
    def __init__(self, encoder):
        super(VideoEncoder, self).__init__()
        self.encoder = encoder  # SimpleMLP(2048, 2048)

    def forward(self, x, length):
        # [ batch, n_clip, channel]

        embeddings = self.encoder(x, length)

        return embeddings

###################
# Resnet
###################

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
