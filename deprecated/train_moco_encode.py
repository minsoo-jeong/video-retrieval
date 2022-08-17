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
from datasets.datasets import VCDBTripletDataset, VCDBTripletDataset_collate_fn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from torch.distributed.optim import DistributedOptimizer
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_args_parsers():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', default=42, type=int)

    # distributed
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


@torch.no_grad()
def evaluate_fivr5k(model, loader, epoch, args):
    model.eval()
    base_model = model.module if args.distributed else model

    fivr = loader.dataset.fivr

    progress = tqdm(loader, ncols=150) if utils.distributed.is_main_process() else None

    features = dict()
    for k, x, l in loader:

        features[k[0]] = base_model.forward_encoder_q(x.cuda(), l.cuda()).cpu().numpy()
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

        output = model(anc.to(args.device),
                       pos.to(args.device),
                       neg.to(args.device), len_a, len_p, len_n)
        loss = criterion(*output)
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


from datasets.datasets import FIVR, VCDB
from utils.array import load_pickle, resize_array


def apply_albumentations_for_video(x, transform):
    aug = transform(image=x[0])['replay']
    x = [transform.replay(aug, image=_x)['image'] for _x in x]

    return torch.stack(x, dim=0)


class FIVRDataset(Dataset):
    def __init__(self, index, version, length=None, transform=None):
        super(FIVRDataset, self).__init__()
        self.fivr = FIVR(version)
        self.index = load_pickle(index)
        self.keys = self.fivr.get_all_keys()
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.index[key]

        x = np.load(path)
        if self.length:
            x, _ = resize_array(x, self.length)
        if self.transform:
            x = apply_albumentations_for_video(x, self.transform)
        return key, x, x.shape[0]


class VCDBTripletDataset(Dataset):
    # anc - pos : pair
    # negative : distraction
    def __init__(self, index, length=64, transform=None):
        self.vcdb = VCDB()
        self.index = load_pickle(index)
        self.length = length
        self.transform = transform

    def __len__(self):
        return len(self.vcdb.pair)

    def __getitem__(self, idx):
        pair = self.vcdb.pair[idx]
        negative = self.vcdb.sampling_negative(1)[0]
        anc_path, pos_path, neg_path = self.index[pair[0]], self.index[pair[1]], self.index[negative]

        anc, pos, neg = np.load(anc_path), np.load(pos_path), np.load(neg_path)

        if self.length:
            anc, _ = resize_array(anc, self.length)
            pos, _ = resize_array(pos, self.length)
            neg, _ = resize_array(neg, self.length)

        if self.transform:
            anc = apply_albumentations_for_video(anc, self.transform)
            pos = apply_albumentations_for_video(pos, self.transform)
            neg = apply_albumentations_for_video(neg, self.transform)

        return anc, pos, neg, anc.shape[0], pos.shape[0], neg.shape[0]


import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.Moco import VideoMoCo

default_transform = A.ReplayCompose([
    A.CenterCrop(224, 224),
    A.Normalize(),
    ToTensorV2()
])


def main(args):
    init_distributed_mode(args)

    set_seed(args.seed)

    model = VideoMoCo(Resnet50(), SimpleMLP(2048, 1024)).to(args.device)

    fivr_dataset = FIVRDataset('/workspace/fivr5k_frames.pkl', version='5k', transform=default_transform)
    train_dataset = VCDBTripletDataset('/workspace/vcdb_frames.pkl', length=16,
                                       transform=default_transform)
    fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=8, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=8,
                              worker_init_fn=seed_worker, shuffle=True, drop_last=True)

    criterion = CircleLoss(m=0.25, gamma=256).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

        fivr_sampler = DistributedSampler(dataset=fivr_dataset, shuffle=False, num_replicas=args.world_size)
        fivr_loader = DataLoader(fivr_dataset, batch_size=1, num_workers=8, sampler=fivr_sampler)

        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, seed=args.seed)
        train_loader = DataLoader(train_dataset,
                                  batch_size=int(64 / args.world_size),
                                  num_workers=8,
                                  worker_init_fn=seed_worker, sampler=train_sampler, drop_last=True)

    # evaluate_fivr5k(model, fivr_loader, 0, args)

    epochs = 10
    for epoch in range(1, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, optimizer, None, epoch, args)
        evaluate_fivr5k(model, fivr_loader, epoch, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parsers()])
    args = parser.parse_args()

    main(args)
