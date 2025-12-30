import torch

'''

'''
a = torch.arange(6).reshape(2, 3)
b = torch.einsum('ij->ji', [a])

print(b)
