import numpy as np
import random

import torch

from utils.array import load_pickle

from datasets import apply_transform

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from PIL import Image


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


# from torchvision.transforms._transforms_video import *
import math, sys


def tt(vid, transform):
    seed = random.randint(0, sys.maxsize)
    # seed = None
    x = []
    for v in vid:
        random.seed(seed)
        torch.manual_seed(seed)
        x.append(transform(Image.fromarray(v)))
        # x.append(transform(image=v)['image'])

    return torch.stack(x, dim=0)


def tt3(vid, transform: A.ReplayCompose) -> torch.tensor:
    aug = transform(image=vid[0])['replay']
    print(aug)
    x = []
    for v in vid:
        x.append(transform.replay(aug, image=v)['image'])
    return torch.stack(x, dim=0)


fivr = load_pickle('/workspace/fivr_frames.pkl')

path = fivr[list(fivr.keys())[1]]
frames = np.load(path)
print(path)

transform = A.Compose([
    A.RandomCrop(100, 224),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
], additional_targets={'image': 'abc'})
from torchvision.transforms import transforms as trn
from torchvision.transforms import functional as F

transform2 = trn.Compose([
    trn.RandomCrop((100, 224)),
    trn.RandomVerticalFlip(p=0.5),
    trn.ToTensor(),
])

transform3 = A.ReplayCompose([
    A.RandomCrop(100, 224),
    A.RandomFog(),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
])

t = tt3(frames, transform3)

# t = tt(frames, transform2)
print('ffff', frames.shape)
print(t.shape)

t = t.permute((0, 2, 3, 1)).numpy()
print(t.shape)

imgs = []
for f in t:
    imgs.append(Image.fromarray(f))

imgss = []
nrow = 4
for n in range(0, len(imgs), nrow):
    im = get_concat_h(imgs[n:n + nrow])
    imgss.append(im)
im = get_concat_v(imgss)

plt.imshow(im)
plt.show()
plt.close()
