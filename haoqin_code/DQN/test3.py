import torch

A = torch.ones(2, 2)
B = torch.zeros(2, 3)

C = torch.cat([A, B], dim=1)
print(C)