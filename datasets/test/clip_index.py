from datasets import load_index, get_clip_indices_from_frames, get_clip_indices_from_video
from fractions import Fraction

if __name__ == '__main__':
    index = load_index('../../data/fivr5k.json')
    k = list(index.keys())[0]
    info = index[k]
    print(info)

    clips, indices = get_clip_indices_from_frames(info, sampling_rate=30, frames_per_clip=3, n_clip=3)
    print(clips, indices)

    clips, start_sec, end_sec, indices = get_clip_indices_from_video(info, sampling_rate=30, frames_per_clip=3, n_clip=3)
    print(clips, start_sec, end_sec, indices)

    sr = 30
    fps = info.get('average_rate')
    print(fps)
    sampling_rate = Fraction(fps / sr)

    print(sampling_rate, float(sampling_rate))

    fps = Fraction(info.get('average_rate'))
    print(fps)
    sampling_rate = Fraction(fps, sr)

    print(sampling_rate, float(sampling_rate))
