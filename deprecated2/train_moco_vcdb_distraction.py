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
from futures.test.models.imac import Resnet_IMAC, Resnet_IRMAC, Resnet_IRMAC_PCA

warnings.filterwarnings(action='ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('train script', add_help=False)

    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--valid_batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

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

    train_transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    simclr_aug = A.ReplayCompose([
        A.ColorJitter(brightness=.8, contrast=.8, saturation=.8, hue=.2, p=0.3),
        A.ToGray(p=.2),
        A.HorizontalFlip(),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(1.0, 2.0), p=0.2),
        A.RandomResizedCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    valid_transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = VideoDataset_VCDBDistraction(vcdb=VCDB(),
                                                 transform_1=simclr_aug,
                                                 transform_2=simclr_aug,
                                                 index='data/vcdb-distraction10k.json',
                                                 frames_per_clip=1,
                                                 sampling_rate=1,
                                                 sampling_unit='fps',
                                                 n_clips=64)

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
    # clip_encoder = ClipEncoder(Resnet50_RMAC())
    # clip_encoder = ClipEncoder(Resnet_IRMAC_PCA(
    #     pca_param='/mldisk/nfs_shared_/dh/graduation_thesis/vcdb/pca_params_vcdb89325_resnet50_rmac_3840.npz'))
    # clip_encoder.requires_grad_(False)
    # video_encoder = VideoEncoder(Transformer(dim=1024, nhead=8, nlayers=1, dropout=0.5))
    # video_encoder = VideoEncoder(SimpleMLP(dim=2048, output_dim=2048))

    # encoder = Encoder(clip_encoder, video_encoder).cuda()

    clip_encoder = Resnet50_IRMAC_TCA_PCA(pca_param='/mlsun/ms/tca/pca/vcdb90k_resnet50_irmac_tca_norm_3840_1024.npz',
                                          n_components=1024,
                                          whitening=True,
                                          normalize=True,
                                          pretrained=True)
    clip_encoder.requires_grad_(False)
    video_encoder = Transformer(dim=1024, nhead=8, nlayers=1, dropout=0.5)
    encoder = Encoder2(clip_encoder, video_encoder).cuda()

    model = MoCo(encoder, dim=1024, K=4096, m=0.99, T=1.).cuda()

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = CircleLoss(m=0.25, gamma=256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = None
    print(args)
    fivr = FIVR('5k')

    for epoch in range(1, 100):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(model, train_loader, criterion, optimizer, scheduler, epoch, args)
        validate(fivr, encoder, valid_loader, epoch, args)

        torch.cuda.empty_cache()


def train(model, loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    losses, progress = None, None
    if is_main_process():
        losses = AverageMeter()
        progress = tqdm(loader, ncols=150)

    for idx, (_, q, k, length) in enumerate(loader, start=1):
        logits, labels = model(q.cuda(), k.cuda(), length.cuda(), length.cuda())

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser('script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:

        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
