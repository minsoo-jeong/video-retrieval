import os
import cv2
import glob
import numpy as np
import pickle as pk
import json
from tqdm import tqdm
from multiprocessing import Pool

from pathlib import Path, PosixPath

import ffmpeg


def get_vcdb_videos(root='/mldisk/nfs_shared_/MLVD/VCDB/videos/'):
    vlist = list(Path(root).rglob('*.mp4'))
    vlist += list(Path(root).rglob('*.flv'))
    return vlist


def get_video_list(root, extension=['*.mp4']):
    videos = []
    for ext in extension:
        videos.extend(list(Path(root).rglob(ext)))
    return videos


def save_frames(src, dst):
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

    try:

        format = '%06d.jpg'
        # min-side <= 360
        stream = (ffmpeg.input(src.absolute().as_posix(), vsync=2, hide_banner=None)
                  .filter('scale', "if(lte(iw, ih), min(iw, 360), -2)", "if(lte(iw, ih),-2,min(ih, 360))")
                  .output(dst.absolute().joinpath(format).as_posix(), **{'q:v': 1})
                  )

        stream.run(quiet=True)

        return 0
    except Exception as e:
        return src.absolute().as_posix()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    root = '/mldisk/nfs_shared_/MLVD/VCDB/videos/distraction'
    videos = sorted(get_video_list(root=root, extension=['*.mp4', '*.flv']))

    print(f'Get {len(videos)} from {root}')

    if args.end == -1:
        args.end = len(videos)
    videos = videos[args.start: args.end]
    print(args, root)
    print(len(videos))

    # videos = [Path("/mldisk/nfs_shared_/MLVD/VCDB/videos/distraction/0034/NfTpqGI5wkg.mp4"),
    #           Path("/mldisk/nfs_shared_/MLVD/VCDB/videos/distraction/0066/kWmGKB6cz8U.mp4")]

    dst_root = '/mlsun/ms/vcdb/frames/distraction'

    pool = Pool(args.pool)
    progress = tqdm(videos)
    results = []
    for path in videos:
        dst = Path(dst_root).joinpath(Path(path).stem)
        # save_frames(path, dst)
        results.append(pool.apply_async(save_frames, args=(path, dst), callback=lambda x: progress.update()))

    pool.close()
    pool.join()

    fail = []
    for r in results:
        r = r.get()
        if r:
            fail.append(r)

    if len(fail):
        with open('fail.json', 'w') as f:
            json.dump(fail, f, indent=2)

    # print([r.get() for r in results])
