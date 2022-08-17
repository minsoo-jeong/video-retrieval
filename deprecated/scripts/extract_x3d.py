import copy
import io

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


class ArrayDataset(Dataset):
    def __init__(self, array):
        super(ArrayDataset, self).__init__()
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]


import tarfile
from utils.utils import show_images, show_images_tensor


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
    def __init__(self, index, frames_per_clip, sampling_rate, transform):
        super(FrameClipsDataset, self).__init__()
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
        clip_info = self.clips[idx].copy()

        print(id(self.clips[idx]), id(clip_info))
        key = clip_info['key']
        frame_dir = Path(self.index[key]['frames'])
        frame_file_format = '{:06d}.jpg'
        frame_index = clip_info['frame_index']

        # print(frame_index)
        frame_names = np.array([frame_file_format.format(idx + 1) for idx in frame_index])

        # print(frame_names)
        # file = tarfile.open(path, 'r')
        # frames = self.load_images_from_archive(file, frame_names)
        # file.close()

        frames = self.load_images_from_directory(frame_dir, frame_names)

        frames = self.apply_transform(frames)

        frames = einops.rearrange(frames, 't c h w -> c t h w')

        clip_info.pop('frame_index')
        print(clip_info, self.clips[idx])
        print(id(clip_info),id(self.clips[idx]))

        exit()

        return clip_info, frames

    def load_images_from_directory(self, root: pathlib.Path, names):
        def _load_image(path, backend: str = 'pillow'):

            if backend == 'cv2':
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif backend == 'pillow':
                image = Image.open(path)
                image = np.asarray(image)
            else:
                raise NotImplementedError(f'Unsupported backend type: {backend}')
            return image

        images = [_load_image(root.joinpath(name), backend='pillow') for name in names]

        return images

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


@torch.no_grad()
def extract(model, frames):
    model.eval()

    loader = DataLoader(ArrayDataset(frames), batch_size=64, shuffle=False)
    features = []
    for frame in loader:
        feature = model(frame.cuda())
        features.append(feature)
    features = torch.cat(features, dim=0)

    return features


@torch.no_grad()
def extract2(model, loader):
    model.eval()

    features = []
    for info, frame in tqdm(loader):
        print(info)
        feature = model(frame.cuda())
        features.append(feature)
    features = torch.cat(features, dim=0)

    return features


from PIL import Image
from tqdm import tqdm

from pytorchvideo.transforms import transforms


def main():
    transform = A.ReplayCompose([
        A.SmallestMaxSize(160),
        A.CenterCrop(160, 160),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = FrameClipsDataset('/workspace/fivr5k_info.json',
                                frames_per_clip=13,
                                sampling_rate=6,
                                transform=transform
                                )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    model = models_3d.TorchHubWrap('x3d_s')
    model = torch.nn.DataParallel(model).cuda()

    extract2(model, loader)

    # for key, frames in tqdm(loader, dynamic_ncols=True):
    #     continue
    # print(frames.shape, key)

    # extract(model, frames[0])

    # print(i.shape)


if __name__ == '__main__':
    main()
