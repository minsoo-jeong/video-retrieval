from pymediainfo import MediaInfo
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

from fractions import Fraction

def parse_mediainfo(path):
    media_info = MediaInfo.parse(path)

    for track in media_info.tracks:
        data = track.to_data()
        print(data.get('track_type'), data.get('duration'),data.get('frame_rate'), Fraction(data.get('frame_rate')),data.get('nb_frame'))
        print(data)
        # if track.track_type == "Video":
        #     print("Bit rate: {t.bit_rate}, Frame rate: {t.frame_rate}, "
        #           "Format: {t.format}".format(t=track)
        #           )
        #     print("Duration (raw value):", track.duration)
        #     print("Duration (other values:")
        #     print(track.other_duration)
        # elif track.track_type == "Audio":
        #     print("Track data:")
        #     print(track.to_data())


def parse_pyav(path):
    vid = EncodedVideo.from_path(path.as_posix(),decode_audio=False)  # EncodedVideoPyAV
    container = vid._container
    stream = vid._container.streams.video[0]
    print(container.duration)
    print(vid.duration, stream.average_rate,stream.frames)

    #


if __name__ == '__main__':
    videos = list(Path('/mlsun/ms/fivr5k/videos').rglob('*.mp4'))

    for v in videos:
        print(v)
        parse_mediainfo(v)

        parse_pyav(v)

