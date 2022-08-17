import pickle as pk
import json


def load_pickle(p):
    with open(p, 'rb') as f:
        content = pk.load(f)
    return content


def load_json(p):
    with open(p, 'r') as f:
        content = json.load(f)
    return content


"""
dns== FIVR != TCA
"""

fivr1 = '/workspace/datasets/fivr.dns.pickle'
fivr2 = '/workspace/datasets/fivr.tca.pickle'
fivr3 = '/workspace/datasets/fivr_annotation.json'

vcdb1 = '/workspace/datasets/vcdb.pkl'
vcdb2 = '/workspace/datasets/vcdb.tca.pkl'

f1 = load_pickle(fivr1)
f2 = load_pickle(fivr2)
f3 = load_json(fivr3)

print(f1.keys())
print(f2.keys())
print(f3.keys())

f1_5k_q = set(f1['5k']['queries'])
f1_5k_d = set(f1['5k']['database'])
f2_5k_q = set(f1['5k']['queries'])
f2_5k_d = set(f1['5k']['database'])
print(len(f1_5k_q.union(f1_5k_d)))
print(len(f2_5k_q.union(f2_5k_d)))

f1_200k_q = set(f1['200k']['queries'])
f1_200k_d = set(f1['200k']['database'])
f2_200k_q = set(f1['200k']['queries'])
f2_200k_d = set(f1['200k']['database'])
print(len(f1_200k_q.union(f1_200k_d)))
print(len(f2_200k_q.union(f2_200k_d)))

db = f1_5k_d
for q in f1_5k_q:
    ann1 = f1['annotation'][q]
    ann2 = f2['annotation'][q]
    ann3 = f3[q]
    print({k: sorted([vv for vv in v if vv in db]) for k, v in sorted(ann1.items()) if k in ['ND', 'DS', 'CS', 'IS']})
    print({k: sorted([vv for vv in v if vv in db]) for k, v in sorted(ann2.items()) if k in ['ND', 'DS', 'CS', 'IS']})
    print({k: sorted([vv for vv in v if vv in db]) for k, v in sorted(ann3.items()) if k in ['ND', 'DS', 'CS', 'IS']})
