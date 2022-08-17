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



def calculate_ap(results, groundtruth):
    c, ap = 0, 0.
    for n, r in enumerate(results, start=1):
        if r in groundtruth:
            c += 1
            ap += c / n

        # find all groundtruth
        if c == len(groundtruth):
            break

    ap /= len(groundtruth)
    return ap
