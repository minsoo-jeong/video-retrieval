import torch


class CircleLoss(torch.nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        alpha = torch.clamp_min(logits + self.m, min=0).detach()
        alpha[labels] = torch.clamp_min(-logits[labels] + 1 + self.m, min=0).detach()
        delta = torch.ones_like(logits, device=logits.device, dtype=logits.dtype) * self.m
        delta[labels] = 1 - self.m
        return self.loss(alpha * (logits - delta) * self.gamma, labels)


from matplotlib import pyplot as plt
from PIL import Image
from typing import List, TypeVar, Type

t_image = TypeVar('Image')


def show_images(images: List[t_image], max: int = 30, nrow: int = 5):
    def concat_vertical(imgs: List[t_image]):
        widths = sum([im.width for im in imgs])
        dst = Image.new('RGB', (widths, imgs[0].height))
        for n, im in enumerate(imgs):
            dst.paste(im, (im.width * n, 0))

        return dst

    def concat_horizontal(imgs: List[t_image]):
        heights = sum([im.height for im in imgs])
        dst = Image.new('RGB', (imgs[0].width, heights))
        for n, im in enumerate(imgs):
            dst.paste(im, (0, im.height * n))

        return dst

    if max > 1:
        images = images[:max]

    vertical = []
    for n in range(0, len(images), nrow):
        vertical.append(concat_vertical(images[n:n + nrow]))
    im = concat_horizontal(vertical)

    plt.imshow(im)
    plt.show()
    plt.close()


import torch
import torchvision.utils as vutils
import numpy as np


def show_images_tensor(images: torch.Tensor, max=32, nrow=5):
    if max > 1:
        images = images[:max]

    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True).cpu()
    im = np.transpose(grid, (1, 2, 0))

    plt.imshow(im)
    plt.show()
    plt.close()


