from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
import argparse
import os

from futures.utils import init_distributed_mode, gather_object, is_main_process, gather_dict, destroy_process_group
from futures.datasets import VideoDataset, FIVR
from futures.models import *


# from futures.test.models.imac import Resnet_IMAC, Resnet_IRMAC, Resnet_IRMAC_PCA


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

    transform = A.ReplayCompose([
        A.SmallestMaxSize(256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = VideoDataset('fivr5k_info.json',
                           frames_per_clip=1,
                           sampling_rate=1,
                           sampling_unit='fps',  # [ 'frames', 'fps']
                           transform=transform
                           )

    fivr = FIVR('5k')
    sampler = None
    if args.distributed:
        args.total_batch_size = args.batch_size * args.world_size
        args.num_workers_per_node = args.num_workers * args.n_proc_per_node
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate,
                        sampler=sampler)
    print(args)

    # clip_encoder = ClipEncoder(TorchHubWrap('x3d_m'))
    # clip_encoder = ClipEncoder(Resnet50_GeM('moco_v3'))
    # clip_encoder = ClipEncoder(Resnet_IRMAC_PCA(
    #     pca_param='/mldisk/nfs_shared_/dh/graduation_thesis/vcdb/pca_params_vcdb89325_resnet50_rmac_3840.npz'))
    # clip_encoder.requires_grad_(False)
    # video_encoder = VideoEncoder(Transformer(dim=2048, nhead=8, nlayers=1, dropout=0.1))
    # video_encoder = VideoEncoder(Basic())
    # encoder = Encoder(clip_encoder, video_encoder).cuda()

    # clip_encoder = Resnet50_GeM('moco_v3')
    # clip_encoder = Resnet50_RMAC()
    # clip_encoder = TorchHubWrap('x3d_m')
    # clip_encoder = Resnet50_IRMAC_PCA(n_components=1024, whitening=True,
    #                                   pca_param='/mlsun/ms/tca/pca/vcdb90k_resnet50_irmac_3840_1024.npz')
    # clip_encoder = Resnet50_IRMAC_TCA_PCA(pca_param='/mlsun/ms/tca/pca/vcdb90k_resnet50_irmac_tca_norm_3840_1024.npz',
    #                                       n_components=1024,
    #                                       whitening=True,
    #                                       normalize=True,
    #                                       pretrained=True)

    clip_encoder = Resnet50_GeM(pretrained=True)
    clip_encoder.requires_grad_(False)
    video_encoder = Basic()
    # video_encoder = Transformer(dim=1024, nhead=8, nlayers=1, dropout=0.5)
    encoder = Encoder(clip_encoder, video_encoder).cuda()

    if args.distributed:
        model_without_ddp = encoder
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        encoder = torch.nn.parallel.DistributedDataParallel(encoder)

    video_features, clip_features = extract(encoder, loader, args)
    if is_main_process():
        map_v = fivr.evaluate(video_features, metric='cosine')
        map_c = fivr.evaluate(clip_features, metric='chamfer')
        print(f'Video-level mAP (DSVR|CSVR|ISVR): {map_v[0]:.4f}|{map_v[1]:.4f}|{map_v[2]:.4f}')
        print(f'Clip-level mAP (DSVR|CSVR|ISVR): {map_c[0]:.4f}|{map_c[1]:.4f}|{map_c[2]:.4f}')


@torch.no_grad()
def extract(model, loader, args):
    model.eval()
    progress = tqdm(loader, ncols=80) if is_main_process() else None
    index_v, index_c = dict(), dict()
    for keys, frames, length in loader:
        vf, cf = model(frames.cuda(), length.cuda(), return_clip_emb=True)

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
        progress.close()
    return index_v, index_c


if __name__ == '__main__':

    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:

        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
