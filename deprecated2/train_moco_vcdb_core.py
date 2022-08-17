from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
import argparse
import os

from futures.utils import *
from futures.datasets import *
from futures.models import *
from futures.losses import *
from futures.moco import MoCo


def get_args_parser():
    parser = argparse.ArgumentParser('train script', add_help=False)

    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--valid_batch_size', default=1, type=int)
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
    args.dist_url = 'tcp://163.239.25.47:32145'

    init_distributed_mode(args)

    train_transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    valid_transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = VideoDataset_VCDBPair(vcdb=VCDB(),
                                          index='data/vcdb-core.json',
                                          frames_per_clip=1,
                                          sampling_rate=1,
                                          sampling_unit='fps',
                                          transform=train_transform,
                                          n_clips=32)

    valid_dataset = VideoDataset(index='data/fivr5k.json',
                                 frames_per_clip=1,
                                 sampling_rate=1,
                                 sampling_unit='fps',  # [ 'frames', 'fps']
                                 transform=valid_transform
                                 )
    train_sampler, valid_sampler = None, None
    if args.distributed:
        args.num_workers_per_node = args.num_workers * args.n_proc_per_node
        args.train_total_batch_size = args.train_batch_size * args.world_size
        args.valid_total_batch_size = args.valid_batch_size * args.world_size
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              num_workers=args.num_workers)

    # clip_encoder = ClipEncoder(TorchHubWrap('x3d_m'))
    clip_encoder = ClipEncoder(Resnet50_RMAC())
    clip_encoder.requires_grad_(False)
    video_encoder = VideoEncoder(Transformer(dim=2048, nhead=8, nlayers=1, dropout=0.3))

    # video_encoder = VideoEncoder(SimpleMLP(dim=2048, output_dim=2048))
    # video_encoder = VideoEncoder(Average())

    encoder = Encoder(clip_encoder, video_encoder).cuda()
    model = MoCo(encoder, dim=2048, K=4096, m=0.999, T=0.07).cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = CircleLoss(m=0.25, gamma=256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    print(args)
    fivr = FIVR('5k')

    for epoch in range(1, 100):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, optimizer, None, epoch, args)
        validate(fivr, encoder, valid_loader, epoch, args)


def train(model, loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    losses, progress = None, None
    if is_main_process():
        losses = AverageMeter()
        progress = tqdm(loader, ncols=150)

    for idx, (_, _, a, p, len_a, len_p) in enumerate(loader, start=1):
        logits, labels = model(a.cuda(), p.cuda(), len_a.cuda(), len_p.cuda())

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_main_process():
            losses.update(loss.item())
            progress.update()
            progress.set_description(f'[Train {epoch:>3}] '
                                     f'Loss: {losses.val:.4f}({losses.avg:.4f}) '
                                     f'LR: {optimizer.param_groups[0]["lr"]:.4e}')

        if scheduler:
            scheduler.step()

    if is_main_process():
        progress.close()
    return losses


@torch.no_grad()
def validate(fivr, encoder, loader, epoch, args):
    encoder.eval()

    # forward = model.forward_q if not args.distributed else model.module.forward_q

    progress = tqdm(loader, ncols=150) if is_main_process() else None

    # extract features
    index_v, index_c = dict(), dict()

    for keys, frames, length in loader:
        vf, cf = encoder(frames.cuda(), length.cuda(), return_clip_emb=True)

        features_v = {k: f.unsqueeze(0).cpu() for k, f in zip(keys, vf)}
        features_c = {k: f[:l].cpu() for k, f, l in zip(keys, cf, length)}

        if args.distributed:
            features_v = gather_dict(features_v, args.rank, args.world_size, dst=0)
            features_c = gather_dict(features_c, args.rank, args.world_size, dst=0)

        if is_main_process():
            index_v.update(features_v)
            index_c.update(features_c)
            progress.update()

    if is_main_process():
        map_c = fivr.evaluate(index_c, metric='chamfer')
        map_v = fivr.evaluate(index_v, metric='cosine')
        progress.set_description(f'[Valid {epoch:>3}] DSVR/CSVR/ISVR: '
                                 f'{map_v[0]:.4f}/{map_v[1]:.4f}/{map_v[2]:.4f}, '
                                 f'{map_c[0]:.4f}/{map_c[1]:.4f}/{map_c[2]:.4f}'
                                 )


@torch.no_grad()
def extract(model, loader, args):
    model.eval()

    index = dict()
    for keys, frames, length in tqdm(loader):
        vf, cf = model(frames.cuda(), length.cuda())

        features = {k: {'video': v.unsqueeze(0),
                        'clip': c[:l]
                        } for k, v, c, l in zip(keys, vf.cpu(), cf.cpu(), length)}

        if args.distributed:
            features = gather_dict(features, args.rank, args.world_size, dst=0)
        if is_main_process():
            index.update(features)

    return index


if __name__ == '__main__':

    parser = argparse.ArgumentParser('script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:

        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
