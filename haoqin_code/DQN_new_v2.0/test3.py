import torch
import numpy as np

A = torch.ones(2, 2)
B = torch.zeros(2, 3)

C = torch.cat([A, B], dim=1)
print(C)

A = torch.zeros(2, 3)
B = torch.ones(2, 2)

B_shape = B.shape

A[0:B_shape[0], 0:B_shape[1]] = B 

print(A)


A = np.ones((2, 3))
B = np.zeros((2, 2))

B_shape = B.shape

A[0:B_shape[0], 0:B_shape[1]] = B 

print(A)