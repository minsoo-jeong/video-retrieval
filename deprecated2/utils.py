import os
import torch
import torch.distributed as dist
from typing import Union, List


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if args.distributed:
        rank = args.local_rank + args.node_rank * args.n_proc_per_node
        world_size = args.n_node * args.n_proc_per_node
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        if args.dist_url == 'env://':
            dist_url = f'{os.environ.get("MASTER_ADDR")}:{os.environ.get("MASTER_PORT")}'
            print(dist_url)
        print('| distributed init (rank {}): {}'.format(rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=world_size, rank=rank)
        torch.distributed.barrier()
        # setup_for_distributed(args.rank == 0)

    args.rank = get_rank()
    args.world_size = get_world_size()


def destroy_process_group():
    dist.destroy_process_group()


def gather_object(obj, rank, world, dst=0) -> List[object]:
    dist.barrier()
    received = [None] * world if rank == dst else None

    dist.gather_object(obj, received, dst=dst)

    return received if rank == dst else None


def gather_dict(obj: dict, rank, world, dst=0):
    received = gather_object(obj, rank, world, dst)  # list of dictionary
    if rank == dst:
        for d in received[1:]:
            received[0].update(d)
        return received[0]
    else:
        return None
