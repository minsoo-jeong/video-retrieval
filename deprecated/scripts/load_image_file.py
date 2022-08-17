from torchvision.datasets.folder import pil_loader

from pathlib import Path
from tqdm import tqdm

from multiprocessing import Pool
import numpy as np
import sys


def load(path):
    for p in path.glob('*.jpg'):
        try:
            img = pil_loader(p)
            img = np.asarray(img)
        except Exception as e:
            print(f'{p}: {e}')


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt

if __name__ == '__main__':
    p = '/mlsun/ms/fivr5k/frames/gfBONrIcJ6k/003584.jpg'
    p2 = '/workspace/help/gfBONrIcJ6k/003584.jpg'

    im = pil_loader(p2)

    plt.imshow(im)

    plt.show()
