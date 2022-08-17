import pytorchvideo.models.net
import torch

from torchsummary import summary
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.data import UniformClipSampler

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
transform_params = {
    "side_size": 186,
    "crop_size": 160,
    "num_frames": 13,
    "sampling_rate": 8,
}
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)

url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
video_path = 'archery.mp4'
try:
    urllib.URLopener().retrieve(url_link, video_path)
except:
    urllib.request.urlretrieve(url_link, video_path)

start_sec = 0
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second
end_sec = start_sec + clip_duration
video = EncodedVideo.from_path(video_path)
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
print(video_data['video'].shape)
video_data = transform(video_data)

inputs = video_data["video"]

print(inputs.shape)

from utils.array import load_pickle

from pytorchvideo.data import kinetics, Hmdb51
from pytorchvideo.data.clip_sampling import ClipInfo, ClipInfoList

import torch.nn as nn
import math
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from fractions import Fraction

fivr = load_pickle('/workspace/fivr5k_videos.pkl')

videos = list(fivr.values())
print(len(videos))


class VideoDataset(nn.Module):
    def __init__(self, paths, n_clips, sampling_rate, frames_per_clip):
        super(VideoDataset, self).__init__()
        self.paths = paths
        self.sampling_rate = sampling_rate
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        video = EncodedVideo.from_path(path, decode_audio=False, decoder='pyav')

        sampler = UniformClipSampler(video.duration, stride=None, backpad_last=False, eps=1e-6)
        clip_info = sampler(0.0, v.duration + 1e-6, None)
        tensor = video.get_clip(clip_info.clip_start_sec, clip_info.clip_end_sec)['video']

        nb_decode_frames = tensor.shape[1] // self.sampling_rate \
            if isinstance(self.sampling_rate, int) else round(video.duration)

        clip = uniform_temporal_subsample(tensor, num_samples=nb_decode_frames, temporal_dim=1)

    def decode(self, video, duration, sampling_rate):
        pass


class VideoDataset2(nn.Module):
    def __init__(self, paths, sampling_rate, frames_per_clip):
        super(VideoDataset2, self).__init__()
        self.paths = paths
        self.sampling_rate = sampling_rate
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        video = EncodedVideo.from_path(path, decode_audio=False, decoder='pyav')
        stream = video._container.streams.video[0]
        duration = video.duration
        if stream.frames == 0:
            nb_frames = video.get_clip(0.0, video.duration + 1e-6)['video'].shape[1]
        else:
            nb_frames = stream.frames
        # transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second
        if isinstance(self.sampling_rate, int):
            sampling_rate = self.sampling_rate
        else:
            sampling_rate = Fraction(nb_frames, duration)  # fps
            print(sampling_rate)
        clip_len = Fraction(self.frames_per_clip * sampling_rate * duration, nb_frames)

        sampler = UniformClipSampler(clip_len, stride=None, backpad_last=True, eps=1e-6)
        clips = []

        info = ClipInfo(0., 0., 0, 0, False)
        while not info.is_last_clip:
            info = sampler(info.clip_end_sec, duration, None)
            clips.append([info, video.get_clip(info.clip_start_sec, info.clip_end_sec)['video'].shape[1]])

        return nb_frames, clips


frames_len = 16
sampling_rate = 'fps'
dataset = VideoDataset2(videos, sampling_rate, frames_per_clip=frames_len)

for nb, i in dataset:
    clip, shape = zip(*i)
    print(nb, sum(shape), shape)

exit()

target_frames = frames_len * sampling_rate
eps = 1e-6
for n, path in enumerate(videos, start=1):
    v = EncodedVideo.from_path(path, decoder="pyav", decode_audio=False)

    stream = v._container.streams.video[0]
    duration = v.duration
    frames_meta = stream.frames
    frames_real = v.get_clip(0, v.duration + eps)['video'].shape
    frames = frames_meta if frames_meta != 0 else frames_real[1]

    target_clip_len = target_frames * duration / frames
    print(f'[{n}/{len(videos)}]', v.name, duration, frames_meta, frames_real, frames, target_clip_len)
    sampler = UniformClipSampler(target_clip_len, stride=None, backpad_last=False, eps=1e-6)

    clips = []
    clips_len = []
    clip = ClipInfo(0., 0., 0, 0, False)
    clip_frames = 0
    while not clip.is_last_clip:
        clip = sampler(clip.clip_end_sec, duration, None)
        f = v.get_clip(clip.clip_start_sec, clip.clip_end_sec)['video'].shape[1]
        clips.append(clip)
        clips_len.append(f)

    print(len(clips), clips)

    print(sum(clips_len), len(clips_len), clips_len)
    print('')
    # sampler = UniformClipSampler(5, stride=None, backpad_last=False, eps=1e-6)
    # start = 0
    #
    # info = sampler.__call__(0, v.duration, None)
    #
    # c = v.get_clip(info.clip_start_sec, info.clip_end_sec)
    # info2 = sampler.__call__(info.clip_end_sec, v.duration, None)
    #
    # total = v.get_clip(0, v.duration)
    #
    # # print(path, v, v.duration,v.get_clip(0,v.duration)['video'].shape)
    #
    # print(path, v.duration, v._duration, c['video'].shape, info, stream.frames, total['video'].shape,
    #       stream.thread_count)
