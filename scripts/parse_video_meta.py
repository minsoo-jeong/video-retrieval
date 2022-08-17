import threading
import time
from pathlib import Path
import re

from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed

from pytorchvideo.data.encoded_video import EncodedVideo

from pymediainfo import MediaInfo
import json

import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from fractions import Fraction


def parse_video(path):
    try:
        video = EncodedVideo.from_path(path)
        stream = video._container.streams.video[0]
        fps = stream.average_rate
        duration = video.duration
        nb_frames = stream.frames

        info = {'video': path.as_posix(),
                'average_rate': float(fps),
                'duration': float(duration),
                'nb_frames': nb_frames,
                }
        return path.stem, info
    except Exception as e:
        print(path, 'EncodedVideo', e)
        return path.stem, {'video': path.as_posix(), 'error': e}


def parse_video_mediainfo(path, frame_path=None):
    try:
        # key, info = parse_video(path)

        media = MediaInfo.parse(path)

        general = media.general_tracks[0].to_data()
        stream = media.video_tracks[0].to_data()

        duration = general.get('duration')
        fps = stream.get('frame_rate')

        nb_frames = stream.get('frame_count', 0)

        if not duration or not fps:
            key, info = parse_video(path)
        else:
            info = {'video': path.as_posix(),
                    'average_rate': float(fps),
                    'duration': float(Fraction(duration, 1000)),
                    'nb_frames': int(nb_frames),
                    }

        if frame_path:
            frame_path = Path(frame_path)
            if frame_path.exists():
                info['frames'] = frame_path.as_posix()
                info['nb_frames'] = len(list(frame_path.glob('*.jpg')))

        return path.stem, info
    except Exception as e:
        print(path, e)
        return path.stem, None


def multithread(videos, frame_root):
    progress = tqdm(videos)
    results = dict()
    fail = []

    pool = ThreadPoolExecutor(max_workers=16)
    workers = [pool.submit(parse_video_mediainfo, v, frame_root.joinpath(v.stem)) for v in videos]

    # pool.map(parse_video, videos)

    for i in as_completed(workers):
        k, v = i.result()
        if v:
            results[k] = v
        else:
            fail.append(k)

        progress.update()

    return results, fail


if __name__ == '__main__':
    videos_root = '/mldisk/nfs_shared_/MLVD/VCDB/videos/distraction'
    frame_root = '/mlsun/ms/vcdb/frames/distraction'
    pattern = re.compile(r'^.*\.(flv|mp4)$')

    videos = sorted([p for p in Path(videos_root).rglob('*') if re.match(pattern, p.name)],
                    key=lambda x: x.stem)

    # multiprocess(videos)
    results, fail = multithread(videos, Path(frame_root))

    json.dump(results, open(f'vcdb-distraction90k.json', 'w'), indent=2)
    if len(fail):
        json.dump(fail, open(f'vcdb-distraction90k-fail.json', 'w'), indent=2)
