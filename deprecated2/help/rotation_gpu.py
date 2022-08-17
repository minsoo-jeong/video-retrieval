import datetime

import cv2
import numpy as np
import multiprocessing as mp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import transforms as trn
from torchvision import models

from tqdm import tqdm
from collections import defaultdict, OrderedDict

import faiss

from pymediainfo import MediaInfo
import subprocess
from PIL import Image
from torchvision.transforms import functional as TF

from typing import Union

import torchvision.utils as vutils

import matplotlib.pyplot as plt


class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


class MobileNet_AVG(nn.Module):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class Segment_AvgPool(nn.Module):
    def __init__(self):
        super(Segment_AvgPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x = torch.mean(x, 1)
        x = self.norm(x)
        return x


class ListDataset(Dataset):
    def __init__(self, l):
        self.l = l
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        im = self.l[idx]
        frame = self.transform(im)
        return frame

    def __len__(self):
        return len(self.l)


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class Rotation(nn.Module):
    def __init__(self):
        super(Rotation, self).__init__()

    def forward(self, x, angle):
        x = TF.rotate(x, angle, expand=False)
        return x


class ListDataset_rotate(Dataset):
    def __init__(self, l):
        self.l = l
        self.transforms = [
            trn.Compose([
                trn.Resize((224, 224)),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])]
        self.transforms += [
            trn.Compose([
                trn.Resize((224, 224)),
                RotationTransform(i),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]) for i in range(90, 360, 90)]

    def __getitem__(self, idx):
        im = self.l[idx]
        images = [tr(im) for tr in self.transforms]
        return images

    def __len__(self):
        return len(self.l)


def decode_video(video, sampling_rate=1, target_size=None):
    media_info = MediaInfo.parse(video)
    metadata = {'file_path': video}
    for track in media_info.tracks:
        if track.track_type == 'General':
            metadata['file_name'] = track.file_name + '.' + track.file_extension
            metadata['file_extension'] = track.file_extension
            metadata['format'] = track.format
        elif track.track_type == 'Video':
            metadata['width'] = int(track.width)
            metadata['height'] = int(track.height)
            metadata['rotation'] = float(track.rotation or 0.)
            metadata['codec'] = track.codec

    frames = []
    w, h = (metadata['width'], metadata['height']) if metadata['rotation'] not in [90, 270] else (
        metadata['height'], metadata['width'])
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-vsync', '2',
               '-i', video,
               '-pix_fmt', 'bgr24',  # color space
               '-vf', f'fps={sampling_rate}',  # '-r', str(decode_rate),
               '-q:v', '0',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        raw_image = pipe.stdout.read(w * h * 3)
        pipe.stdout.flush()
        try:
            image = Image.frombuffer('RGB', (w, h), raw_image, "raw", 'BGR', 0, 1)
        except ValueError as e:
            break

        if target_size is not None:
            image = TF.resize(image, target_size)
        frames.append(image)
    return frames


@torch.no_grad()
def extract_feature_rotate(model, aggr_model, video, batch=4, seg=5, progress=False):
    model.eval()
    frames = decode_video(video)
    loader = DataLoader(ListDataset_rotate(frames), batch_size=batch, shuffle=False, num_workers=4)
    if progress:
        bar = tqdm(loader, ncols=200)

    frame_features = [[], [], [], []]
    for i, images in enumerate(loader):
        for n in range(4):
            frame_features[n].append(model(images[n].cuda()))

        if progress:
            bar.update()
    frame_features = [torch.cat(f) for f in frame_features]
    if progress:
        bar.close()

    seg_features = [[], [], [], []]
    for n, feature in enumerate(frame_features):
        for ff in torch.split(feature, seg):
            out = aggr_model(ff.unsqueeze(0)).cpu()
            seg_features[n].append(out)
        seg_features[n] = torch.cat(seg_features[n]).numpy()

    return seg_features


@torch.no_grad()
def extract_feature_rotate_gpu(model, aggr_model, video, batch=4, seg=5, progress=False):
    model.eval()
    frames = decode_video(video)
    loader = DataLoader(ListDataset(frames), batch_size=batch, shuffle=False, num_workers=4)
    pbar = tqdm(loader, ncols=200) if progress else None

    rotate_gpu = Rotation().cuda()

    frame_features = [[], [], [], []]  # [origin, rotate 90, rotate 180, rotate 270]
    for i, images in enumerate(loader):
        # origin
        images = images.cuda()
        frame_features[0].append(model(images))

        # rotate 90
        r_90 = rotate_gpu(images, 90)
        frame_features[1].append(model(r_90))

        # rotate 180
        r_180 = rotate_gpu(r_90, 90)
        frame_features[2].append(model(r_180))

        # rotate 270
        r_270 = rotate_gpu(r_180, 90)
        frame_features[3].append(model(r_270))

        if pbar:
            pbar.update()

    frame_features = [torch.cat(f) for f in frame_features]
    if pbar:
        pbar.close()

    seg_features = [[], [], [], []]  # [origin, rotate 90, rotate 180, rotate 270]
    for n, feature in enumerate(frame_features):
        for ff in torch.split(feature, seg):
            out = aggr_model(ff.unsqueeze(0)).cpu()
            seg_features[n].append(out)
        seg_features[n] = torch.cat(seg_features[n]).numpy()
    return seg_features


@torch.no_grad()
def test(video, batch):
    ##############
    # show sample
    ##############
    frames = decode_video(video)
    loader = DataLoader(ListDataset(frames), batch_size=batch, num_workers=0)
    rotate_gpu = Rotation().cuda()
    model = MobileNet_AVG().cuda()

    sample = next(iter(loader))
    plt.figure(dpi=600)
    plt.suptitle('gpu rotation')

    sample = sample.cuda()
    draw_img_tensor(sample.cpu(), (1, 4, 1), "R 0")

    r90 = rotate_gpu(sample, 90)
    draw_img_tensor(r90.cpu(), (1, 4, 2), "R 90")

    r180 = rotate_gpu(r90, 90)
    draw_img_tensor(r180.cpu(), (1, 4, 3), "R 180")

    r270 = rotate_gpu(r180, 90)
    draw_img_tensor(r270.cpu(), (1, 4, 4), "R 270")

    plt.show()

    ##########################################
    # Check GPU utilize: watch -n .2 nvidia-smi
    ##########################################
    for it in range(10):
        for n, sample in enumerate(loader):
            sample = sample.cuda()
            model(sample)
            r90 = rotate_gpu(sample, 90)
            model(r90)
            r180 = rotate_gpu(r90, 90)
            model(r180)
            r270 = rotate_gpu(r180, 90)
            model(r270)

            print(it, n, sample.device, r90.device, r180.device, r270.device, sample.shape)


def test2(video, batch):
    ##########################################
    # Compare Rotation Image CPU vs GPU
    ##########################################
    frames = decode_video(video)

    loader_gpu = DataLoader(ListDataset(frames), batch_size=batch, shuffle=False, num_workers=4)
    rotate_gpu = Rotation().cuda()

    sample = next(iter(loader_gpu)).cuda()
    r90 = rotate_gpu(sample, 90)
    r180 = rotate_gpu(r90, 90)
    r270 = rotate_gpu(r180, 90)
    sample_gpu = [sample.cpu(), r90.cpu(), r180.cpu(), r270.cpu()]

    loader_cpu = DataLoader(ListDataset_rotate(frames), batch_size=batch, shuffle=False, num_workers=4)
    sample_cpu = next(iter(loader_cpu))

    plt.figure(dpi=600)
    plt.suptitle('rotation CPU vs GPU')

    draw_img_tensor(sample_cpu[0], (2, 4, 1), "CPU R 0")
    draw_img_tensor(sample_cpu[1], (2, 4, 2), "CPU R 90")
    draw_img_tensor(sample_cpu[2], (2, 4, 3), "CPU R 180")
    draw_img_tensor(sample_cpu[3], (2, 4, 4), "CPU R 270")

    draw_img_tensor(sample_gpu[0], (2, 4, 5), "GPU R 0")
    draw_img_tensor(sample_gpu[1], (2, 4, 6), "GPU R 90")
    draw_img_tensor(sample_gpu[2], (2, 4, 7), "GPU R 180")
    draw_img_tensor(sample_gpu[3], (2, 4, 8), "GPU R 270")

    plt.show()

    for i_cpu, i_gpu in zip(sample_cpu, sample_gpu):
        print(f'Average Image difference : {np.mean(np.abs(i_cpu.numpy() - i_gpu.numpy()))}')


def draw_img_tensor(im_tensor, grid, title):
    plt.subplot(*grid)
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(im_tensor, nrow=4, padding=2, normalize=True), (1, 2, 0)))


if __name__ == '__main__':
    video = 'tango_01.mp4'

    test(video, batch=32)
    test2(video, batch=32)

    model = MobileNet_AVG().cuda()
    # model.load_state_dict('...')
    model.eval()

    aggr_model = Segment_AvgPool().cuda()
    aggr_model.eval()

    # GPU
    features_gpu = extract_feature_rotate_gpu(model, aggr_model, video, batch=4, seg=5, progress=True)

    # CPU
    features_cpu = extract_feature_rotate(model, aggr_model, video, batch=4, seg=5, progress=True)

    for f_cpu, f_gpu in zip(features_cpu, features_gpu):
        print(f'Average Feature difference : {np.mean(np.abs(f_cpu - f_gpu))}')
