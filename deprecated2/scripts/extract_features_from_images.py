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

from futures.datasets import load_index
from futures.models import *
from futures.utils import init_distributed_mode, gather_object, is_main_process, gather_dict, destroy_process_group


class ImageDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        image = self.transform(image=np.asarray(default_loader(path)))['image']
        return path, image


def get_args_parser():
    parser = argparse.ArgumentParser('extraction script', add_help=False)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--n_node', default=1, type=int,
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

    frames = load_index('../data/vcdb-90k-10frames.json')

    transform = A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = ImageDataset(frames, transform)
    sampler = None
    model = Resnet50_RMAC().cuda()

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    loader = DataLoader(dataset,
                        batch_size=64,
                        num_workers=4,
                        sampler=sampler
                        )

    model.eval()
    progress = tqdm(loader, ncols=150) if is_main_process() else None
    features = []
    with torch.no_grad():
        for _, images in loader:
            f = model(images.cuda()).cpu()

            if args.distributed:
                batch = gather_object(f, args.rank, args.world_size, dst=0)

            if is_main_process():
                progress.update()
                features.append(torch.cat(batch))

    if is_main_process():
        progress.close()
        features = torch.cat(features)
        print(features.shape)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:

        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
