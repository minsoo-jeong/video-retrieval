import torchvision.utils as vutils

import numpy as np

import matplotlib.pyplot as plt

draw_img_tensor(sample.cpu(), (1, 4, 1), "R 0")

def show_tensors(tensors):
    plt.figure(dpi=600)
    plt.suptitle('gpu rotation')


def draw_img_tensor(im_tensor, grid, title):
    plt.subplot(*grid)
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(im_tensor, nrow=4, padding=2, normalize=True), (1, 2, 0)))
