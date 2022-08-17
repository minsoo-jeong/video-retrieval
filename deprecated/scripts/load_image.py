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



if __name__ == '__main__':

    root = Path('/mlsun/ms/fivr5k/frames')

    directories = sorted(list(root.glob('g*')))
    directories += sorted(list(root.glob('G*')))

    pbar = tqdm(directories)

    pool = Pool(16)
    ret = []
    for r in directories:

        ret.append(pool.apply_async(load, args=(r,), callback=lambda x: pbar.update()))


    pool.close()
    pool.join()

    # for r in ret:
    #     rr = r.get()

    #
    # for path in pbar:
    #     try:
    #         pil_loader(path)
    #     except Exception as e:
    #         pbar.write(f'{path}: {e}')
    #
    #     pbar.set_description(f'{path}')
    #     pbar.update()
