import numpy as np
import pickle as pk
import faiss
import json


def load_pickle(path):
    with open(path, 'rb') as f:
        content = pk.load(f)
    return content


def load_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def resize_array(x, size, pad_value=0, sampling_method='sequential'):
    """
    Pad/Sampling Array
    """
    if x.shape[0] < size:
        x, size = pad_array(x, size, value=pad_value)
    elif x.shape[0] > size:
        x = sampling_array(x, size, method=sampling_method, index=False)

    return x, size


def pad_array(x, size, value=0):
    assert x.shape[0] < size
    shape = x.shape
    pad_shape = (size - shape[0], *shape[1:])
    pad = np.zeros(pad_shape) + value
    pad = pad.astype(x.dtype)
    array = np.concatenate([x, pad], axis=0)

    return array, shape[0]


def sampling_array(x, size, method='sequential', index=False):
    assert x.shape[0] > size

    length = x.shape[0]

    if method.lower() == 'sequential':
        start = np.random.randint(0, length - size)
        idx = np.arange(start, start + size, step=1, dtype=int)
    elif method.lower() == 'uniform':
        idx = np.linspace(0, length, size, endpoint=False, dtype=int)
    elif method.lower() == 'random':
        idx = np.arange(0, length, step=1, dtype=int)
        idx = np.sort(np.random.choice(idx, size, replace=False))
    elif method.lower() == 'mix':
        idx = np.arange(0, length, step=1, dtype=int)
        idx = np.random.choice(idx, size, replace=False)
    else:
        raise NotImplemented()

    array = x[idx]

    if index:
        array = (array, idx)
    return array


def similarity_search(query, database):
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    dist, idx = index.search(query, k=database.shape[0])
    return dist, idx
