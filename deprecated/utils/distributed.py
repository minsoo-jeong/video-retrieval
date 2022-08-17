import torch
import torch.distributed as dist
import os


# From deit repository

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.distributed:
        rank = args.local_rank + args.node_rank * args.n_proc_per_node
        world_size = args.n_node * args.n_proc_per_node
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=world_size, rank=rank)
        torch.distributed.barrier()
        # setup_for_distributed(args.rank == 0)

    args.rank = get_rank()
    args.world_size = get_world_size()


def gather_dict(dict, rank, world_size):
    dist.barrier()
    all_dict = [None] * world_size
    dist.all_gather_object(all_dict, dict)
    if rank == 0:
        for d in all_dict:
            dict.update(d)
    else:
        all_dict, dict = None, None
    return dict


def gather_object(obj, rank, world, dst=0):
    dist.barrier()
    received = [None] * world if rank == dst else None

    dist.gather_object(obj, received, dst=dst)

    return received if rank == dst else None
