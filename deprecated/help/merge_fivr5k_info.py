import json

from pathlib import Path


def load_json_files(files):
    d = dict()
    for f in files:
        with open(f, 'r') as fp:
            d.update(json.load(fp))
    return d


files = list(Path('/workspace/data/fivr5k').glob('fivr5k_info-*.json'))
info = load_json_files(files)

print(len(list(info.keys())))
# print(info)

for k, v in info.items():

    if v['nb_frames'] == 0:
        nb_frame = v['average_rate'] * v['duration']

        info[k]['nb_frames'] = int(nb_frame)
    if v['nb_frames'] != v['frame_archive_length'] - 1:
        info[k]['nb_frames'] = v['frame_archive_length'] - 1

    info[k]['frames'] = f'/mlsun/ms/fivr5k/frames/{k}'

with open('/workspace/fivr5k_info.json', 'w') as f:
    json.dump(info, f, indent=2)
