import random
from pathlib import Path
import cv2
from PIL import Image
import tarfile
import numpy as np
import io
from datetime import datetime
import os


def time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        elapsed = datetime.now() - start
        # print(f'Elapsed Time for {func.__name__}, {args, kwargs}: {elapsed.total_seconds():.4f} sec')
        return result, elapsed

    return wrapper


def _load_image_archive(archive: tarfile.TarFile, member: tarfile.TarInfo, backend: str = 'cv2'):
    file = archive.extractfile(member)
    if backend == 'cv2':
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == 'pillow':
        image = Image.open(file)
        image = np.asarray(image)

    else:
        return NotImplemented(f'Not supported backend type: {backend}')
    return image


def _load_image_from_path(path, backend='cv2'):
    if backend == 'cv2':
        image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif backend == 'pillow':
        image = Image.open(path)
        image = np.asarray(image)
    else:
        return NotImplemented(f'Not supported backend type: {backend}')
    return image


@time
def load_images_from_archive(archive, n_images=1, backend='cv2'):
    file = tarfile.open(archive, 'r')
    names = [f for f in file.getnames() if f.endswith('jpg')]
    names = sorted(random.choices(names, k=n_images))

    images = [_load_image_archive(file, file.getmember(name), backend=backend) for name in names]

    return images


@time
def load_images_from_directory(directory, n_images=1, backend='cv2'):
    paths = [f for f in directory.glob('*.jpg')]
    paths = sorted(random.choices(paths, k=n_images))

    images = [_load_image_from_path(path, backend=backend) for path in paths]
    return images


from utils.utils import show_images

if __name__ == '__main__':
    archive_root = Path('/mlsun/ms/fivr5k/frames-tar')
    frame_root = Path('/mlsun/ms/fivr5k/frames')

    count = 100
    n_images = 100

    archives = sorted([arc for arc in archive_root.rglob('*.tar')])[:count]
    directories = sorted([d for d in frame_root.iterdir()])[:count]
    t1 = 0.
    for n, arc in enumerate(archives):
        images, elapsed = load_images_from_archive(arc, n_images=n_images, backend='pillow')
        t1 += elapsed.total_seconds()
        if n == 0:
            print(images[0].shape)
        #     show_images(images)
    print(t1)

    t2 = 0.
    for directory in directories:
        images, elapsed = load_images_from_directory(directory, n_images=n_images, backend='pillow')
        t2 += elapsed.total_seconds()
    print(t2)

    t3 = 0.
    for arc in archives:
        images, elapsed = load_images_from_archive(arc, n_images=n_images, backend='cv2')
        t3 += elapsed.total_seconds()
    print(t3)

    t4 = 0.
    for directory in directories:
        images, elapsed = load_images_from_directory(directory, n_images=n_images, backend='cv2')
        t4 += elapsed.total_seconds()

    print(t4)
