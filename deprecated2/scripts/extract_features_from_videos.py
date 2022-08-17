import math

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader, default_loader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.multiprocessing as mp

import argparse

from futures.datasets import VideoDataset
from futures.utils import init_distributed_mode, gather_object, is_main_process, gather_dict, destroy_process_group
from futures.models import Resnet50_IRMAC_TCA_PCA, ClipEncoder

from futures.datasets import FIVR

from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('extraction script', add_help=False)

    # distributed training parameters
    parser.add_argument('--n_node', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--n_proc_per_node', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--node_rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main_worker(gpu, ngpus_per_node, args):
    args.local_rank = gpu
    args.dist_url = 'tcp://163.239.27.244:32145'

    init_distributed_mode(args)
    target_dir = Path('/mlsun/ms/tca/features/vcdb90k-resnet50-irmac-tca-norm-pca')
    transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = VideoDataset('../data/vcdb-90k.json',
                           frames_per_clip=1,
                           sampling_rate=1,
                           sampling_unit='fps',  # [ 'frames', 'fps']
                           transform=transform
                           )
    sampler = None
    model = ClipEncoder(
        Resnet50_IRMAC_TCA_PCA(pca_param='/mlsun/ms/tca/pca/vcdb90k_resnet50_irmac_tca_norm_3840_1024.npz',
                               n_components=1024,
                               whitening=True,
                               normalize=True,
                               pretrained=True)).cuda()

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=dataset.collate,
                        num_workers=4,
                        sampler=sampler)

    model.eval()
    progress = tqdm(loader, ncols=150) if is_main_process() else None
    fivr = FIVR('5k')
    index = dict()
    with torch.no_grad():
        for key, frames, clips in loader:
            c = math.ceil(clips[0].item() / 64)
            features = []
            for f in torch.chunk(frames, c, dim=1):
                length = torch.tensor([f.shape[1]])
                feat = model(f.cuda(), length.cuda()).cpu()
                features.append(feat)
            features = {key[0]: torch.cat(features, dim=1).squeeze(0)}

            np.save(target_dir.joinpath(key[0]).with_suffix('.npy').as_posix(), features[key[0]].numpy())

            if args.distributed:
                features = gather_dict(features, args.rank, args.world_size, dst=0)

            if is_main_process():
                progress.update()
                index.update(features)

        if is_main_process():
            progress.close()
            map_c = fivr.evaluate(index, metric='chamfer')
            print(f'Clip-level mAP (DSVR|CSVR|ISVR): {map_c[0]:.4f}|{map_c[1]:.4f}|{map_c[2]:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:

        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
