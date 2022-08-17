import time


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args)
        elapsed = time.time() - start
        print(f'time : {elapsed}')
        return result

    return wrapper


@time_it
def add(*args):
    return sum(args)


if __name__ == '__main__':
    r = add(1, 2, 3, 4, 5)
    print(r)
