import torch


def string_to_byte_tensor(string, encode='utf8'):
    tensor = torch.ByteTensor(list(bytes(string, encode)))
    return tensor


a = 'text'

print(string_to_byte_tensor(a))

b = [a] * 5
print(b)
