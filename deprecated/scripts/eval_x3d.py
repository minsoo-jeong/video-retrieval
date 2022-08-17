import io

import cv2
import pytorchvideo.transforms
import torch
import numpy as np
from datasets.datasets import FIVRDataset, FIVR
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
from fractions import Fraction

from pytorchvideo.data.kinetics import Kinetics
from pytorchvideo.data import UniformClipSampler
from pytorchvideo.data.clip_sampling import ClipInfo

from datetime import datetime

timer1 = 0.
timer2 = 0.


class VideoDataset(Dataset):
    def __init__(self, index, frames_per_clip, sampling_rate, transform, sampling_rate_unit='frames'):
        super(VideoDataset, self).__init__()
        self.index = load_pickle(index)
        self.keys = sorted(list(self.index.keys()))
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.sampling_rate_unit = sampling_rate_unit
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        path = self.index[key]
        video = EncodedVideo.from_path(path, decode_audio=False, decoder='pyav')

        global timer1, timer2
        decode_start = datetime.now()
        all_frames = video.get_clip(0.0, video.duration + 1e-6)['video']
        timer1 += (datetime.now() - decode_start).total_seconds()

        sampling_rate = self.sampling_rate
        if self.sampling_rate_unit.lower() == 'fps':  # sampling N frames per second
            sampling_rate = round(Fraction(all_frames.shape[1], video.duration * self.sampling_rate))

        nb_clips = round(all_frames.shape[1] / (self.frames_per_clip * sampling_rate))

        frame_index = torch.arange(0, nb_clips * self.frames_per_clip) * sampling_rate
        frame_index = torch.clamp(frame_index, 0, all_frames.shape[1] - 1).long()
        frames = torch.index_select(all_frames, dim=1, index=frame_index)

        transform_start = datetime.now()
        frames = self.apply_transform(frames)  # c t h w -> t c h w
        frames = einops.rearrange(frames, '(n t) c h w -> n c t h w', n=nb_clips, t=self.frames_per_clip)
        timer2 += (datetime.now() - transform_start).total_seconds()

        return frames

    def apply_transform(self, frames):
        frames = einops.rearrange(frames, 'c t h w -> t h w c').numpy()
        aug = self.transform(image=frames[0])['replay']
        frames = [self.transform.replay(aug, image=f)['image'] for f in frames]
        frames = torch.stack(frames, dim=0)
        return frames

    def _show(self, frames, max_frames=30, nrow=4):

        def get_concat_h(imgs):
            widths = sum([im.width for im in imgs])
            dst = Image.new('RGB', (widths, imgs[0].height))
            for n, im in enumerate(imgs):
                dst.paste(im, (im.width * n, 0))

            return dst

        def get_concat_v(imgs):
            heights = sum([im.height for im in imgs])
            dst = Image.new('RGB', (imgs[0].width, heights))
            for n, im in enumerate(imgs):
                dst.paste(im, (0, im.height * n))

            return dst

        frames = einops.rearrange(frames, 'n c t h w -> (n t) h w c').numpy().astype(np.uint8)[:max_frames]

        imgs = []
        for t in frames:
            imgs.append(Image.fromarray(t))
        imgss = []
        for n in range(0, len(imgs), nrow):
            im = get_concat_h(imgs[n:n + nrow])
            imgss.append(im)
        im = get_concat_v(imgss)

        plt.imshow(im)
        plt.show()
        plt.close()


class ClipDataset(Dataset):
    def __init__(self, index, frames_per_clip, sampling_rate, transform, sampling_rate_unit='frames'):
        super(ClipDataset, self).__init__()
        self.index = load_pickle(index)
        self.keys = sorted(list(self.index.keys()))[:100]
        self.frames_per_clip = frames_per_clip
        self.sampling_rate = sampling_rate
        self.sampling_rate_unit = sampling_rate_unit
        self.transform = transform

        self.clips = self.load_clips()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        key, info = self.clips[idx]
        path = self.index[key]

        video = EncodedVideo.from_path(path, decode_audio=False, decoder='pyav')

        global timer1, timer2
        decode_start = datetime.now()
        frames = video.get_clip(info.clip_start_sec, info.clip_end_sec)['video']
        timer1 += (datetime.now() - decode_start).total_seconds()

        frame_index = torch.linspace(0, frames.shape[1] - 1, self.frames_per_clip).long()
        frames = torch.index_select(frames, dim=1, index=frame_index)  # c t h w

        transform_start = datetime.now()
        frames = self.apply_transform(frames)
        timer2 += (datetime.now() - transform_start).total_seconds()

        return frames

    def load_clips(self):
        total_clips = []
        for key in tqdm(self.keys):
            path = self.index[key]
            video = EncodedVideo.from_path(path, decode_audio=False, decoder='pyav')

            stream = video._container.streams.video[0]
            fps = stream.average_rate
            if fps is None:
                all_frames = video.get_clip(0, video.duration + 1e-6)['video']
                fps = Fraction(all_frames.shape[1], video.duration)

            sampling_rate = self.sampling_rate
            if self.sampling_rate_unit.lower() == 'fps':  # sampling N frames per second
                sampling_rate = round(Fraction(fps, self.sampling_rate))

            clip_sec = Fraction(self.frames_per_clip * sampling_rate, fps)
            sampler = UniformClipSampler(clip_sec, stride=None, backpad_last=False, eps=1e-6)
            info = ClipInfo(0., 0., 0, 0, False)
            while not info.is_last_clip:
                info = sampler(info.clip_end_sec, video.duration, None)
                total_clips.append((key, info))
        return total_clips

    def apply_transform(self, frames):
        frames = einops.rearrange(frames, 'c t h w -> t h w c').numpy()
        aug = self.transform(image=frames[0])['replay']
        frames = [self.transform.replay(aug, image=f)['image'] for f in frames]
        frames = torch.stack(frames, dim=0)  # t c h w
        frames = einops.rearrange(frames, 't c h w -> c t h w')
        return frames


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
                image = Image.open(io.BytesIO(file.read()))
            else:
                return NotImplemented(f'Not supported backend type: {backend}')
            return image

        images = [_load_image(archive, archive.getmember(name), backend='cv2') for name in names]

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
def extract_clip(model, loader):
    global timer1, timer2
    model.eval()

    features = []
    for frame in tqdm(loader):
        print(timer1, timer2)
        feature = model(frame.cuda())
        # features.append(feature)
    # features = torch.cat(features, dim=0)

    return features


import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from pytorchvideo.transforms import transforms


def main():
    # v_transform = pytorchvideo.transforms.ApplyTransformToKey({    })
    transform = A.ReplayCompose([
        A.SmallestMaxSize(160),
        A.CenterCrop(160, 160),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = ArchiveFramesDataset('/workspace/fivr5k_frames_tar.json',
                                   frames_per_clip=13,
                                   sampling_rate=30,
                                   transform=transform
                                   )
    # for i in dataset:
    #     print(i.shape)
    #
    # exit()

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    model = models_3d.TorchHubWrap('x3d_s')
    model = torch.nn.DataParallel(model).cuda()

    # with torch.no_grad():
    #     for i in tqdm(range(100)):
    #         data = torch.rand((64, 3, 16, 312, 312))
    #         model(data.cuda())

    # extract_clip(model, loader)

    for key, frames in tqdm(loader):
        extract(model, frames[0])

    # print(i.shape)


if __name__ == '__main__':
    main()
