import copy
import math
import numpy as np
# import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models

from utils.distributed import is_dist_avail_and_initialized, get_world_size

import einops


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=1024, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        # if use hvd
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, a, p, n, len_a, len_p, len_n):
        """
        Input:
            a: a batch of anchor logits
            p: a batch of positive logits
            n: a bigger batch of negative logits
        Output:
            logits, targets
        """

        if len(n.size()) > 3:
            n = n.view(-1, n.size()[2], n.size()[3])
            len_n = len_n.view(-1, 1)

        # compute query features
        q = self.encoder_q(a, len_a)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            p = self.encoder_k(p, len_p)  # anchors: NxC
            p = F.normalize(p, dim=1)
            k = self.encoder_k(n, len_n)  # keys: kNxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, p]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_encoder_q(self, x, len_x):
        return self.encoder_q(x, len_x)


class VideoMoCo(nn.Module):
    def __init__(self, frame_encoder, base_encoder, dim=1024, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(VideoMoCo, self).__init__()
        self.frame_encoder = frame_encoder
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue

        # if use hvd
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def encode_frames(self, x):
        b, t, *_ = x.shape

        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.frame_encoder(x)

        x = einops.rearrange(x, '(b t) c -> b t c', b=b, t=t)
        return x

    def forward(self, a, p, n, len_a, len_p, len_n):
        """
        Input:
            a: a batch of anchor logits
            p: a batch of positive logits
            n: a bigger batch of negative logits
        Output:
            logits, targets
        """
        with torch.no_grad():
            a = self.encode_frames(a)
            p = self.encode_frames(p)
            n = self.encode_frames(n)

        # compute query features
        q = self.encoder_q(a, len_a)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            p = self.encoder_k(p, len_p)  # anchors: NxC
            p = F.normalize(p, dim=1)
            k = self.encoder_k(n, len_n)  # keys: kNxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, p]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_encoder_q(self, x, len_x):
        return self.encoder_q(x, len_x)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if hvd.is_initialized():
    #     return hvd.allgather(tensor.contiguous())
    if is_dist_avail_and_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        return torch.cat(tensors_gather, dim=0)
    else:
        return tensor
