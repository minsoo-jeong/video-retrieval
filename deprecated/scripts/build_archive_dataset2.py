from pathlib import Path
import json
import tarfile

from pytorchvideo.data.encoded_video import EncodedVideo
from multiprocessing import Pool
from tqdm import tqdm
import pickle as pk
from fractions import Fraction

def parse(key):
    video = video_root.joinpath(key + '.mp4').as_posix()
    frame_arc = arc_root.joinpath(key + '.tar').as_posix()
    vid = EncodedVideo.from_path(video)
    stream = vid._container.streams.video[0]
    fps = stream.average_rate
    duration = vid.duration
    nb_frames = stream.frames

    arc = tarfile.open(frame_arc)
    arc_frames = len(arc.getnames()) - 1

    info = {'video': video,
            'average_rate': float(fps),
            'duration': float(duration),
            'nb_frames': stream.frames,
            'frame_archive': frame_arc,
            'frame_archive_length': len(arc.getnames()),
            }
    arc.close()
    vid.close()
    return info


video_root = Path('/mlsun/ms/fivr5k/videos')
arc_root = Path('/mlsun/ms/fivr5k/frames-tar')
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--pool', type=int, default=8)
    args = parser.parse_args()

    videos = sorted(list(video_root.rglob('*.mp4')))
    if args.end == -1:
        args.end = len(videos)

    print(args)
    print(len(videos))

    videos = videos[args.start: args.end]
    info = dict()

    progress = tqdm(videos)

    pool = Pool(args.pool)
    for v in videos:
        key = v.stem

        info[key] = pool.apply_async(parse, [key], callback=lambda *x: progress.update())
        # info[key] = parse(v.as_posix(), frame_arc.as_posix())
        # progress.update()
    pool.close()
    pool.join()

    for k, v in info.items():
        info[k] = v.get()

    print(info)

    json = json.dump(info,open(f'/workspace/fivr5k_info-{args.start}_{args.end}.json', 'w'), indent=2)
