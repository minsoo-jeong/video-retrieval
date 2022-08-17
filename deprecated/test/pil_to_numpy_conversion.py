from PIL import Image
import numpy as np

import time


def elapsed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f'Function {"[" + func.__name__ + "]":^10} elapsed time: {elapsed:.4f}')
        return result, elapsed

    return wrapper


@elapsed
def array(image):
    return np.array(image)


@elapsed
def asarray(image):
    return np.asarray(image)


def random_image(size):
    rand = np.random.rand(*size, 3) * 255
    return Image.fromarray(rand.astype(np.uint8))


"""
size: 1920*1920
test: 100
nsarray | array
0.5590  | 0.6623

size: 1280*1280
test: 100
nsarray | array
0.2256  | 0.2630

size: 1280*720
test: 100
nsarray | array
0.1542  | 0.1264

size: 720*720
test: 100
nsarray | array
0.0574  | 0.0888

size: 256*256
test: 100
nsarray | array
0.0089  | 0.0080

size: 100*100
test: 100
nsarray | array
0.0024  | 0.0023
"""
if __name__ == '__main__':
    size = (256, 256)
    test = 100
    t1 = .0
    for _ in range(test):
        pil_image = random_image(size)
        arr2, t = asarray(pil_image)
        t1 += t

    t2 = .0
    for _ in range(test):
        pil_image = random_image(size)
        arr, t = array(pil_image)
        t2 += t

    print('[nsarray]', t1)
    print('[array]', t2)
