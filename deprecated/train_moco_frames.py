import argparse
import os

import numpy as np
import torch

import utils.distributed
from utils.distributed import *
from utils.seed import *
from models.Moco import MoCo
from models.models import *
from datasets.datasets import FIVRDataset, VCDBTripletDataset, VCDBTripletDataset_collate_fn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class Encoder(torch.nn.Module):
    def __init__(self, base_model, output_dim, batch_size):
        super(Encoder, self).__init__()
        self.encoder = base_model
        self.batch_size = batch_size
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        x, n, t = self.pre(x)
        x = self.encode(x)
        x = self.post(x, n, t)

        return x

    def pre(self, x):
        n, t, *_ = x.shape
        x = einops.rearrange(x, 'n t c h w -> (n t) c h w')
        return x, n, t

    def post(self, x, n, t):
        x = einops.rearrange(x, '(n t) c -> n t c', n=n, t=t)
        return x

    @torch.no_grad()
    def encode(self, x):
        encodings = torch.zeros((x.shape[0], self.output_dim), device=x.device)
        for n, batch in enumerate(torch.split(x, self.batch_size)):
            code = self.encoder(batch)
            encodings[n * self.batch_size:n * self.batch_size + code.shape[0]] = code

        return encodings

    @torch.no_grad()
    def encode2(self, x):
        encodings = []
        for batch in torch.split(x, self.batch_size):
            encodings.append(batch)
        return torch.cat(encodings)


class CircleLoss(torch.nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        alpha = torch.clamp_min(logits + self.m, min=0).detach()
        alpha[labels] = torch.clamp_min(-logits[labels] + 1 + self.m, min=0).detach()
        delta = torch.ones_like(logits, device=logits.device, dtype=logits.dtype) * self.m
        delta[labels] = 1 - self.m
        return self.loss(alpha * (logits - delta) * self.gamma, labels)


def get_args_parsers():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', default=42, type=int)

    # distributed
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


@torch.no_grad()
def evaluate_fivr5k(model, loader, epoch, args):
    model.eval()
    fivr = loader.dataset.fivr
    progress = tqdm(loader, ncols=150)

    features = dict()
    for k, x, l in loader:
        features[k[0]] = model(x.cuda(), l).cpu().numpy()
        progress.update()

    if args.distributed:
        dist.barrier()
        all_features = [None] * args.world_size
        dist.all_gather_object(all_features, features)
        if args.rank == 0:
            for d in all_features:
                features.update(d)

    if utils.distributed.is_main_process():
        dsvr, csvr, isvr = fivr.evaluate(features)
        progress.set_description(f'[Epoch {epoch} Eval] '
                                 f'DSVR: {dsvr:.4f}, CSVR: {csvr:.4f}, ISVR: {isvr:.4f}')

    progress.close()


@torch.no_grad()
def evaluate_fivr5k_encode(encoder, model, loader, epoch, args):
    encoder.eval()
    model.eval()
    fivr = loader.dataset.fivr
    progress = tqdm(loader, ncols=150)

    features = dict()
    for k, x, l in loader:
        n, t, *_ = x.shape
        x = encoder(x.to(args.device))
        features[k[0]] = model(x, l).cpu().numpy()
        progress.update()

    if args.distributed:
        dist.barrier()
        all_features = [None] * args.world_size
        dist.all_gather_object(all_features, features)
        if args.rank == 0:
            for d in all_features:
                features.update(d)

    if utils.distributed.is_main_process():
        dsvr, csvr, isvr = fivr.evaluate(features)
        progress.set_description(f'[Epoch {epoch} Eval] '
                                 f'DSVR: {dsvr:.4f}, CSVR: {csvr:.4f}, ISVR: {isvr:.4f}')

    progress.close()


def main(args):
    init_distributed_mode(args)

    set_seed(args.seed)
    print(args)

    model = SimpleMLP(2048, 2048).to(args.device)
    encoder = Resnet50().to(args.device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    ddp_encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], output_device=args.gpu)

    fivr_dataset = FIVRDataset('/workspace/fivr_frames.pickle', version='5k')
    if args.distributed:
        fivr_sampler = DistributedSampler(dataset=fivr_dataset, shuffle=False, num_replicas=args.world_size)
        fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=8, sampler=fivr_sampler)

    else:
        fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=8, shuffle=False)

    evaluate_fivr5k_encode(ddp_encoder,ddp_model, fivr_loader, 0, args)

    # train_dataset = VCDBTripletDataset('/workspace/vcdb_resnet50-avgpool-mlsun.pkl', n_negative=1,
    #                                    length=32, )
    # if args.distributed:
    #     train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, seed=args.seed)
    #     train_loader = DataLoader(train_dataset,
    #                               collate_fn=VCDBTripletDataset_collate_fn,
    #                               batch_size=64,
    #                               num_workers=8,
    #                               worker_init_fn=seed_worker,
    #                               sampler=train_sampler,
    #                               drop_last=True)
    # else:
    #     train_loader = DataLoader(train_dataset,
    #                               collate_fn=VCDBTripletDataset_collate_fn,
    #                               batch_size=64,
    #                               shuffle=True,
    #                               num_workers=8,
    #                               worker_init_fn=seed_worker,
    #                               drop_last=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parsers()])
    args = parser.parse_args()

    main(args)
