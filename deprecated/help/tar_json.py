import json

from pathlib import Path

root = Path('/mlsun/ms/fivr5k/frames-tar')

tarfiles = root.rglob('*.tar')
ff = dict()
for f in sorted(tarfiles):
    k = f.stem
    v = f.as_posix()
    ff[k] = v

with open('/workspace/fivr5k_frames_tar.json', 'w') as f:
    json.dump(ff, f, indent=2)
