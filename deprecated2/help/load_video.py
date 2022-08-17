import einops
import math
import numpy as np
from fractions import Fraction
from pytorchvideo.data.encoded_video import EncodedVideo
import torch

info = {
    "-0NokAd5WSY": {
        "video": "/mlsun/ms/fivr5k/videos/-0NokAd5WSY.mp4",
        "average_rate": 29.97002997002997,
        "duration": 54.054,
        "nb_frames": 1620,
        "frame_archive": "/mlsun/ms/fivr5k/frames-tar/-0NokAd5WSY.tar",
        "frame_archive_length": 1621,
        "frames": "/mlsun/ms/fivr5k/frames/-0NokAd5WSY"
    },
    "-1SnlEMzOi4": {
        "video": "/mlsun/ms/fivr5k/videos/-1SnlEMzOi4.mp4",
        "average_rate": 24.88888888888889,
        "duration": 72.05151927437642,
        "nb_frames": 1792,
        "frame_archive": "/mlsun/ms/fivr5k/frames-tar/-1SnlEMzOi4.tar",
        "frame_archive_length": 1793,
        "frames": "/mlsun/ms/fivr5k/frames/-1SnlEMzOi4"
    },
    "55dde762dc283ee301065f670b6d7c08c3c32668": {
        "video": "/mldisk/nfs_shared_/MLVD/VCDB/videos/core_dataset/president_obama_takes_oath/55dde762dc283ee301065f670b6d7c08c3c32668.flv",
        "average_rate": 30.0,
        "duration": 149.833,
        "nb_frames": 4495
    }
}


def load_video_from_frame(info, sampling_rate=30, frames_per_clip=1, n_clip=None):
    len_clip = frames_per_clip * sampling_rate
    clips = max(round(info['nb_frames'] / len_clip), 1)

    offset, n_frames = 0, clips * frames_per_clip

    if n_clip:
        n_frames = n_clip * frames_per_clip
        clips = min(clips, n_clip)
        if info['nb_frames'] > len_clip * clips:
            offset = np.random.choice(info['nb_frames'] - len_clip * clips)  # random offset

    indices = np.arange(0, n_frames) * sampling_rate + offset
    indices = np.clip(indices, offset, info['nb_frames'])
    print(clips, offset, indices, indices.shape)


def load_video_from_path(info, sampling_rate=30, frames_per_clip=1, n_clip=None):
    len_clip = frames_per_clip * sampling_rate
    fps = Fraction(info['average_rate'])
    nb_frames = math.floor(info['duration'] * fps)
    clips = max(round(nb_frames / len_clip), 1)

    offset, n_frames = 0, clips * frames_per_clip
    if n_clip:
        n_frames = n_clip * frames_per_clip
        clips = min(clips, n_clip)
        if nb_frames > len_clip * clips:
            offset = np.random.choice(nb_frames - len_clip * clips)

    start_sec = Fraction(offset, fps)
    end_sec = start_sec + Fraction(clips * len_clip, fps)

    video = EncodedVideo.from_path(info['video'], decode_audio=False, decoder='pyav')
    frames = video.get_clip(start_sec, end_sec + 1e-6)['video']

    indices = np.arange(0, n_frames) * sampling_rate

    print(start_sec,frames.shape, indices, n_frames, n_clip * sampling_rate)

    indices = torch.tensor(np.clip(indices, 0, frames.shape[1] - 1))

    frames = torch.index_select(frames, dim=1, index=indices)

    frames = einops.rearrange(frames, 'c t h w -> t c h w')

    print(clips, frames.shape)


info = list(info.values())

# load_video_from_frame(info[0], sampling_rate=16, frames_per_clip=5, n_clip=3)
load_video_from_path(info[0], sampling_rate=8, frames_per_clip=8, n_clip=8)
