import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def cosine_faiss_dist(a, b):
    na = a / np.linalg.norm(a, ord=2, axis=1, keepdims=True)
    nb = b / np.linalg.norm(b, ord=2, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(nb.shape[1])
    index.add(nb)
    dist, idx = index.search(na, k=nb.shape[0])

    e_dist = 1 - np.take_along_axis(dist, np.argsort(idx, axis=1), axis=1)
    print(e_dist)
    return e_dist


def cosine_scipy_dist(a, b):
    e_dist = cdist(a, b, metric='cosine')
    print(e_dist)
    return e_dist


def cosine_einops_dist(a, b):
    na = F.normalize(torch.tensor(a), p=2, dim=1)
    nb = F.normalize(torch.tensor(b), p=2, dim=1)

    e_dist = 1 - torch.einsum('ik,jk->ij', [na, nb])
    print(e_dist)
    return e_dist


def l2_faiss_dist(a, b):
    index = faiss.IndexFlatL2(b.shape[1])
    index.add(b)
    dist, idx = index.search(a, k=b.shape[0])

    e_dist = np.take_along_axis(dist, np.argsort(idx, axis=1), axis=1) ** 0.5
    print(e_dist)
    return e_dist


def l2_scipy_dist(a, b):
    e_dist = cdist(a, b, metric='euclidean')
    print(e_dist)
    return e_dist


def cosine_similarity(a, b, backend='einops'):
    if backend == 'scipy':
        dist = 1 - cdist(a, b, metric='cosine')
    elif backend == 'faiss':
        na = a / np.linalg.norm(a, ord=2, axis=1, keepdims=True)
        nb = b / np.linalg.norm(b, ord=2, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(nb.shape[1])
        index.add(nb)
        dist, idx = index.search(na, k=nb.shape[0])

    elif backend == 'einops':
        na = F.normalize(torch.tensor(a), p=2, dim=1)
        nb = F.normalize(torch.tensor(b), p=2, dim=1)
        dist = torch.einsum('ik,jk->ij', [na, nb])
    else:
        raise NotImplementedError
    return dist


def cosine_distance(a, b, backend='einops'):
    return 1 - cosine_similarity(a,b,backend)


from functools import partial


def chamfer_distance(a, b, compare=partial(cosine_similarity, backend='einops')):
    def _chamfer(a, b, compare):
        dist = compare(a, b)
        dist = dist.max(dim=1)[0].sum().cpu().item() / dist.shape[0]
        return dist

    for q in a:
        for d in b:
            print(_chamfer(q, d, compare))

    dist = np.array([[_chamfer(q, d, compare) for d in b] for q in a])

    return dist


import time

if __name__ == '__main__':

    # np.random.seed(42)
    # a = np.random.random((2, 10, 5)).astype(np.float32)
    # b = np.random.random((3, 20, 5)).astype(np.float32)
    # print('start')
    #
    # dist = chamfer_distance(a, a)
    # print(dist)
    # exit()
    # t1 = 0
    # for q in a:
    #     start = time.time()
    #     for r in b:
    #         cosine_distance(q, r, backend='einops')
    #     t1 += time.time() - start
    # print(t1)
    # exit()

    # for i in range(5):
    #     a = np.random.random((100, 2048)).astype(np.float32)
    #     b = np.random.random((100, 2048)).astype(np.float32)
    #     t1 = time.time()
    #     cosine_distance(a, b, backend='scipy')
    #     t2 = time.time()
    #     cosine_distance(a, b, backend='faiss')
    #     t3 = time.time()
    #     cosine_distance(a, b, backend='einops')
    #     t4 = time.time()
    #
    #     print(t2 - t1, t3 - t2, t4 - t3)

    a = np.random.random((3, 5)).astype(np.float32) * 10
    b = np.random.random((2, 5)).astype(np.float32) * 10
    print(a, b)

    cosine_faiss_dist(a, b)
    cosine_scipy_dist(a, b)
    cosine_einops_dist(a, b)

    l2_faiss_dist(a, b)
    l2_scipy_dist(a, b)
