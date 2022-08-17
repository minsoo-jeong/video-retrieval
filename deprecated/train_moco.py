import argparse
import os

import numpy as np
import torch

import utils.distributed
from utils.distributed import *
from utils.seed import *
from utils.measure import *
from utils.utils import *
from models.Moco import MoCo
from models.models import *
from datasets.datasets import FIVRDataset, VCDBTripletDataset, VCDBTripletDataset_collate_fn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch.distributed.optim import DistributedOptimizer
import sys


def get_args_parsers():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', default=42, type=int)

    # distributed
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


@torch.no_grad()
def evaluate_fivr5k(model, loader, epoch, args):
    model.eval()
    model_without_ddp = model.module if args.distributed else model

    fivr = loader.dataset.fivr

    progress = tqdm(loader, ncols=150) if utils.distributed.is_main_process() else None

    features = dict()
    for k, x, l in loader:

        features[k[0]] = model_without_ddp.forward_encoder_q(x.cuda(), l).cpu().numpy()
        if utils.distributed.is_main_process():
            progress.update()

    if args.distributed:
        features = gather_dict(features, args.rank, args.world_size)

    if utils.distributed.is_main_process():
        dsvr, csvr, isvr = fivr.evaluate(features)
        progress.set_description(f'[Eval {epoch:>4}] '
                                 f'DSVR: {dsvr:.4f}, CSVR: {csvr:.4f}, ISVR: {isvr:.4f}')

        progress.close()


def train(model, loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    losses = AverageMeter()

    progress = tqdm(loader, ncols=150) if utils.distributed.is_main_process() else None

    for anc, pos, neg, len_a, len_p, len_n in loader:
        optimizer.zero_grad()

        output, target = model(anc.cuda(),
                               pos.cuda(),
                               neg.cuda(),
                               len_a,
                               len_p,
                               len_n)
        loss = criterion(output, target)
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        if utils.distributed.is_main_process():
            progress.update()
            progress.set_description(f'[Train {epoch:>3}] '  # [iter {epoch * len(loader) + i}]
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) LR: {optimizer.param_groups[0]["lr"]:.4e}')

        if scheduler is not None:
            scheduler.step()

    if utils.distributed.is_main_process():
        progress.close()


def main(args):
    init_distributed_mode(args)

    set_seed(args.seed)

    model = SimpleMLP(2048, 2048).cuda()
    model = MoCo(model, dim=2048, K=4096, m=0.999, T=0.07).cuda()

    fivr_dataset = FIVRDataset('/workspace/fivr_resnet50-avgpool-mlsun.pkl', version='5k')
    train_dataset = VCDBTripletDataset('/workspace/vcdb_resnet50-avgpool-mlsun.pkl', n_negative=1,
                                       length=32, )
    fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=4, shuffle=False)
    train_loader = DataLoader(train_dataset, collate_fn=VCDBTripletDataset_collate_fn, batch_size=64, num_workers=4,
                              worker_init_fn=seed_worker, shuffle=True, drop_last=True)
    criterion = CircleLoss(m=0.25, gamma=256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

        fivr_sampler = DistributedSampler(dataset=fivr_dataset, shuffle=False, num_replicas=args.world_size)
        fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=4, sampler=fivr_sampler)

        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_dataset, collate_fn=VCDBTripletDataset_collate_fn,
                                  batch_size=64,
                                  num_workers=4,
                                  worker_init_fn=seed_worker, sampler=train_sampler, drop_last=True)

    evaluate_fivr5k(model, fivr_loader, 0, args)

    epochs = 10
    for epoch in range(1, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, optimizer, None, epoch, args)
        evaluate_fivr5k(model, fivr_loader, epoch, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parsers()])
    args = parser.parse_args()
    args.distributed = False
    main(args)
