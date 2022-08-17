import os
import cv2
import glob
import numpy as np
import pickle as pk
import json
from tqdm import tqdm
from multiprocessing import Pool


# Min side Max length
def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frame, desired_size):
    old_size = frame.shape[:2]
    top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
    left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
    return frame[top: top + desired_size, left: left + desired_size, :]


def load_video(video):
    cv2.setNumThreads(1)  # edited

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25

    i = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            if i == 0 or int((i + 1) % round(fps)) == 0:
                # frames.append(center_crop(resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 256), 256))
                frames.append(resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 256))
        else:
            break
        i = i + 1
    cap.release()
    if i > 0:
        frames.append(frames[-1])

    return np.array(frames)


def load_video_paths_ccweb(root='~/datasets/CC_WEB_VIDEO/'):
    paths = sorted(glob.glob(root + 'Videos/*/*.*'))
    vid2paths = {}
    for path in paths:
        vid2paths[path.split('/')[-1].split('.')[0]] = path
    return vid2paths


def load_video_paths_vcdb(root='~/datasets/vcdb/'):
    paths = sorted(glob.glob(root + 'core_dataset/*/*.*'))
    vid2paths_core = {}
    for path in paths:
        vid2paths_core[path.split('/')[-1].split('.')[0]] = path

    paths = sorted(glob.glob(root + 'background_dataset/*/*.*'))
    vid2paths_bg = {}
    for path in paths:
        vid2paths_bg[path.split('/')[-1].split('.')[0]] = path
    return vid2paths_core, vid2paths_bg


def get_frames_ccweb(vid2paths, root='~/datasets/CC_WEB_VIDEO/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        np.save(root + 'Frames/' + vid + '.npy', frames)


def get_frames_vcdb_core(vid2paths, root='~/datasets/vcdb/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        if not os.path.exists(root + 'frames/core/'):
            os.mkdir(root + 'frames/core/')
        np.save(root + 'frames/core/' + vid + '.npy', frames)


def get_frames_vcdb_bg(vid2paths, root='~/datasets/vcdb/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        if not os.path.exists(root + 'frames/background_dataset/' + path.split('/')[-2] + '/'):
            os.mkdir(root + 'frames/background_dataset/' + path.split('/')[-2] + '/')
        np.save(root + 'frames/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)


def load_video_paths_evve(root='~/datasets/evve/'):
    paths = sorted(glob.glob(root + 'videos/*/*.mp4'))
    vid2paths = {}
    for path in paths:
        vid2paths[path.split('/')[-1].split('.')[0]] = path
    return vid2paths


def f(args):
    vid, path = args
    frames = load_video(path)
    np.save('~/datasets/CC_WEB_VIDEO/Frames/' + vid + '.npy', frames)


def g(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('~/datasets/vcdb/frames/core/'):
        os.mkdir('~/datasets/vcdb/frames/core/')
    np.save('~/datasets/vcdb/frames/core/' + vid + '.npy', frames)


def h(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('~/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/'):
        os.mkdir('~/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/')
    np.save('~/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)


from pathlib import Path, PosixPath


def get_vcdb_videos(root='/mldisk/nfs_shared_/MLVD/VCDB/videos/'):
    vlist = list(Path(root).rglob('*.mp4'))
    vlist += list(Path(root).rglob('*.flv'))
    return vlist


def save_frames(src, dst):
    if isinstance(src, PosixPath):
        src = src.as_posix()
    frames = load_video(src)

    np.save(dst, frames)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--pool', type=int, default=8)
    args = parser.parse_args()

    root = '/mldisk/nfs_shared_/MLVD/FIVR/videos'
    videos = sorted(get_vcdb_videos(root=root))
    if args.end == -1:
        args.end = len(videos)
    videos = videos[args.start: args.end]
    print(args, root)
    print(len(videos))

    dst_root = '/mldisk/nfs_shared_/ms/fivr/frames/'
    pool = Pool(args.pool)
    progress = tqdm(videos)
    for path in videos:
        dst = Path(dst_root).joinpath(Path(path).stem + '.npy').as_posix()
        pool.apply_async(save_frames, ((path, dst)), callback=lambda x: progress.update())
    pool.close()
    pool.join()
