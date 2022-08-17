import io
import argparse
import itertools

import cv2
import pathlib
import pytorchvideo.transforms
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from utils.array import load_pickle, resize_array, load_json
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo

import albumentations as A
from albumentations.pytorch import ToTensorV2
import einops

default_transform = A.ReplayCompose([
    A.SmallestMaxSize(182),
    A.CenterCrop(182, 182),
    A.Normalize(),
    ToTensorV2()
])

from models import models_3d

import tarfile
from utils.utils import show_images, show_images_tensor

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.datasets.folder import pil_loader
import copy
from typing import NamedTuple, Union, List


class ClipInfo(NamedTuple):
    key: Union[str, List[str]]
    clip_index: Union[int, List[int]]
    is_last_clip: Union[bool, List[bool]]


class ArchiveFramesDataset(Dataset):
    def __init__(self, index, frames_per_clip, sampling_rate, transform):
        super(ArchiveFramesDataset, self).__init__()
        self.index = load_json(index)
        self.keys = sorted(list(self.index.keys()))
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.index[key]
        file = tarfile.open(path, 'r')
        frame_names = sorted([f for f in file.getnames() if f.endswith('jpg')])

        nb_frames = len(frame_names)
        nb_clips = round(nb_frames / (self.frames_per_clip * self.sampling_rate))

        frame_index = torch.arange(0, nb_clips * self.frames_per_clip) * self.sampling_rate
        frame_index = torch.clamp(frame_index, 0, nb_frames - 1)

        frames = self.load_images_from_archive(file, [frame_names[i] for i in frame_index])

        frames = self.apply_transform(frames)

        frames = einops.rearrange(frames, '(n t) c h w -> n c t h w', t=self.frames_per_clip)

        file.close()

        return key, frames

    def load_images_from_archive(self, archive: tarfile.TarFile, names: list):
        def _load_image(archive: tarfile.TarFile, member: tarfile.TarInfo, backend: str = 'pillow'):
            file = archive.extractfile(member)
            if backend == 'cv2':
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif backend == 'pillow':
                image = Image.open(file)
                image = np.asarray(image)
            else:
                raise NotImplementedError(f'Unsupported backend type: {backend}')
            return image

        images = [_load_image(archive, archive.getmember(name), backend='pillow') for name in names]

        return images

    def apply_transform(self, frames):
        aug = self.transform(image=frames[0])['replay']
        frames = [self.transform.replay(aug, image=f)['image'] for f in frames]

        frames = torch.stack(frames, dim=0)

        return frames


class ArchiveFrameClipsDataset(Dataset):
    def __init__(self, index, frames_per_clip, sampling_rate, transform):
        super(ArchiveFrameClipsDataset, self).__init__()
        self.index = load_json(index)
        self.keys = sorted(list(self.index.keys()))
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.transform = transform

        self.clips = self.load_clips()

    def __len__(self):
        return len(self.clips)

    def load_clips(self):
        clips = []
        for key in tqdm(self.keys):
            info = self.index[key]
            nb_frames = info['nb_frames']
            nb_clips = round(nb_frames / (self.frames_per_clip * self.sampling_rate))

            frame_index = np.arange(0, nb_clips * self.frames_per_clip) * self.sampling_rate
            frame_index = np.clip(frame_index, 0, nb_frames - 1).reshape((nb_clips, self.frames_per_clip))

            clips.extend([{'key': key,
                           'clip_index': n,
                           'frame_index': index,
                           'is_last_clip': n + 1 == len(frame_index)}
                          for n, index in enumerate(frame_index)])

        return clips

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        key = clip_info['key']
        path = self.index[key]['frame_archive']
        frame_file_format = '{}/{:06d}.jpg'
        frame_index = clip_info['frame_index']

        # print(frame_index)
        frame_names = np.array([frame_file_format.format(key, idx + 1) for idx in frame_index])

        # print(frame_names)
        file = tarfile.open(path, 'r')

        frames = self.load_images_from_archive(file, frame_names)
        file.close()

        frames = self.apply_transform(frames)

        return key, frames

    def load_images_from_archive(self, archive: tarfile.TarFile, names: list):
        def _load_image(archive: tarfile.TarFile, member: tarfile.TarInfo, backend: str = 'pillow'):
            file = archive.extractfile(member)
            if backend == 'cv2':
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif backend == 'pillow':
                image = Image.open(file)
                image = np.asarray(image)
            else:
                raise NotImplementedError(f'Unsupported backend type: {backend}')
            return image

        images = [_load_image(archive, archive.getmember(name), backend='pillow') for name in names]

        return images

    def apply_transform(self, frames):
        aug = self.transform(image=frames[0])['replay']
        frames = [self.transform.replay(aug, image=f)['image'] for f in frames]

        frames = torch.stack(frames, dim=0)

        return frames


class FrameClipsDataset(Dataset):
    frame_name_format = '{:06d}.jpg'

    def __init__(self, index, frames_per_clip, sampling_rate, transform):
        super(FrameClipsDataset, self).__init__()
        self.index = load_json(index)
        self.keys = sorted(list(self.index.keys()))
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.transform = transform

        # self.frame_file_format = '{:06d}.jpg'

        self.clips = self.load_clips()

    def __len__(self):
        return len(self.clips)

    def load_clips(self):
        clips = []
        for key in tqdm(self.keys):
            info = self.index[key]
            nb_frames = info['nb_frames']
            nb_clips = max(round(nb_frames / (self.frames_per_clip * self.sampling_rate)), 1)

            frame_index = np.arange(0, nb_clips * self.frames_per_clip) * self.sampling_rate
            frame_index = np.clip(frame_index, 0, nb_frames - 1).reshape((nb_clips, self.frames_per_clip))

            clips.extend([(ClipInfo(key, n, n + 1 == len(frame_index)), index) for n, index in enumerate(frame_index)])

        return clips

    def __getitem__(self, idx):

        clip, frame_index = self.clips[idx]

        key = clip.key
        frame_dir = Path(self.index[key]['frames'])

        names = np.array([self.frame_name_format.format(idx + 1) for idx in frame_index])

        frames = self.load_images_from_directory(frame_dir, names)

        frames = self.apply_transform(frames)

        frames = einops.rearrange(frames, 't c h w -> c t h w')

        return clip, frames

    def load_images_from_directory(self, root: pathlib.Path, names):
        def _load_image(path, backend: str = 'pillow'):
            if backend == 'cv2':
                image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            elif backend == 'pillow':
                image = np.asarray(pil_loader(path))
            else:
                raise NotImplementedError(f'Unsupported backend type: {backend}')
            return image

        images = [_load_image(root.joinpath(name), backend='pillow') for name in names]

        return images

    def apply_transform(self, frames):
        aug = self.transform(image=frames[0])['replay']
        frames = [self.transform.replay(aug, image=f)['image'] for f in frames]

        frames = torch.stack(frames, dim=0)

        return frames

    def get_clip(self, key):
        info = self.index[key]
        nb_frames = info['nb_frames']
        nb_clips = max(round(nb_frames / (self.frames_per_clip * self.sampling_rate)), 1)

        frame_index = np.arange(0, nb_clips * self.frames_per_clip) * self.sampling_rate
        frame_index = np.clip(frame_index, 0, nb_frames - 1).reshape((nb_clips, self.frames_per_clip))

        clips, indices = list(
            zip(*[(ClipInfo(key, n, n + 1 == len(frame_index)), idx) for n, idx in enumerate(frame_index)]))
        return clips, indices

    def get_frames(self, clip, frame_index):
        key = clip.key
        frame_dir = Path(self.index[key]['frames'])

        names = np.array([self.frame_name_format.format(idx + 1) for idx in frame_index])

        frames = self.load_images_from_directory(frame_dir, names)

        frames = self.apply_transform(frames)

        frames = einops.rearrange(frames, 't c h w -> c t h w')

        return clip, frames


from torch.utils.data._utils.collate import default_collate


def collate_clipinfo(clips):
    key, index, is_last = [], [], []

    for clip in clips:
        key.extend(clip.key)
        index.extend(clip.clip_index)
        is_last.extend(clip.is_last_clip)

    return ClipInfo(key, torch.tensor(index), torch.tensor(is_last))


def uncollate_features(clipinfo, all_features):
    keys = np.unique(clipinfo.key)
    features = dict()
    clips = dict()

    for k in keys:
        index = np.where(clipinfo.key == np.array([k]))[0]  # find same video index

        clip_ids = clipinfo.clip_index[index]  # get same video clips

        # sorted_index = index[np.argsort(clip_ids)]
        sorted_index = np.take(index, np.argsort(clip_ids))  # index (sorted by clip index)

        clip_ids_s = clipinfo.clip_index[sorted_index]

        clip_ids, indices = np.unique(clip_ids_s, return_index=True)  # non duplicated clip index

        index = torch.tensor(sorted_index[indices]).long()

        clip = ClipInfo(k, clipinfo.clip_index[index], clipinfo.is_last_clip[index])
        features[k] = torch.index_select(all_features, dim=0, index=index)  # video features
        clips[k] = clip

    feat_len = [v.shape[0] for k, v in features.items()]
    print(feat_len, sum(feat_len))

    return features


@torch.no_grad()
def extract(model, loader, args):
    model.eval()

    clips, features = [], []
    for clip, frame in tqdm(loader):
        feature = model(frame.cuda()).cpu()

        if args.distributed:
            feature = gather_object(feature, args.rank, args.world_size, dst=0)
            clip = gather_object(clip, args.rank, args.world_size, dst=0)

            if is_main_process():
                feature = torch.cat(feature, dim=0)
                clip = collate_clipinfo(clip)

        clips.append(clip)
        features.append(feature)

    # features.append(feature.cpu())

    if is_main_process():
        clips = collate_clipinfo(clips)
        features = torch.cat(features, dim=0)

        print(features.shape)
        features = uncollate_features(clips, features)

    return features


from PIL import Image
from tqdm import tqdm

from pytorchvideo.transforms import transforms
from utils.distributed import *
import random


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main_worker(gpu, ngpus_per_node, args):
    args.local_rank = gpu
    args.dist_url = 'tcp://163.239.27.247:32145'

    init_distributed_mode(args)
    set_seed(42)
    print(args)

    transform = A.ReplayCompose([
        A.SmallestMaxSize(312),
        A.CenterCrop(312, 312),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset = FrameClipsDataset('/workspace/fivr5k_info.json',
                                frames_per_clip=16,
                                sampling_rate=5,
                                transform=transform
                                )

    batch_size = 64
    num_worker = 8
    sampler = None

    if args.distributed:
        batch_size = int(batch_size / args.world_size)
        num_worker = int((num_worker + args.n_proc_per_node - 1) / args.n_proc_per_node)  # ceil(num_worker/n_proc)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=False)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                        sampler=sampler)

    model = models_3d.TorchHubWrap('x3d_l').cuda()

    if args.distributed:
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(model)

    features = extract(model, loader, args)

    if is_main_process():

        root = Path('/mlsun/ms/fivr5k/x3d_l_16_5_312/')
        for k, feature in features.items():
            dst = root.joinpath(f'{k}.npy')
            np.save(dst.as_posix(), feature.numpy())


import os


def get_args_parser():
    parser = argparse.ArgumentParser('extraction script', add_help=False)

    # distributed training parameters
    parser.add_argument('--n_node', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--n_proc_per_node', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--node_rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


import torch.multiprocessing as mp
import warnings

if __name__ == '__main__':

    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()

    args.distributed = args.n_node * args.n_proc_per_node > 1

    if args.distributed:
        mp.spawn(main_worker, nprocs=args.n_proc_per_node, args=(args.n_proc_per_node, args))

    else:
        main_worker(0, args.n_proc_per_node, args)
