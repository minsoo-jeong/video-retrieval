import json

with open('vcdb-90k.json', 'r') as f:
    data = json.load(f)
    print(data)
    print(list(data.keys()).__len__())

    for k, d in data.items():
        if not d.get('nb_frames'):
            print(d)
