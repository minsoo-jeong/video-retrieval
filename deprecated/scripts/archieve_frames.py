import os
import cv2
import glob
import numpy as np
import pickle as pk
import json
from tqdm import tqdm
from multiprocessing import Pool


from pathlib import Path, PosixPath


def get_vcdb_videos(root='/mldisk/nfs_shared_/MLVD/VCDB/videos/'):
    vlist = list(Path(root).rglob('*.mp4'))
    vlist += list(Path(root).rglob('*.flv'))
    return vlist


def get_frame_dir_list(root):
    frame_dirs = [p for p in Path(root).iterdir()]

    return frame_dirs


import ffmpeg

import tarfile


def archive_frames(directory, target):

    with tarfile.open(target, 'w') as tf:
        tf.add(directory.as_posix(), arcname=directory.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--pool', type=int, default=8)
    args = parser.parse_args()

    root = '/mlsun/ms/fivr5k/frames'
    videos = sorted(get_frame_dir_list(root=root))
    print(f'Get {len(videos)} from {root}')

    if args.end == -1:
        args.end = len(videos)
    videos = videos[args.start: args.end]
    print(args, root)
    print(len(videos))

    dst_root = '/mlsun/ms/fivr5k/frames-tar/'

    pool = Pool(args.pool)
    progress = tqdm(videos)
    for path in videos:
        dst = Path(dst_root).joinpath(Path(path).stem + '.tar')
        pool.apply_async(archive_frames, ((path, dst)), callback=lambda x: progress.update())
        # progress.update()
    pool.close()
    pool.join()
