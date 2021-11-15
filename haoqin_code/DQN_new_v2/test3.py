import torch
import numpy as np
import os

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

A = np.ones(shape=(2, 2))
# A = np.expand_dims(A, axis=0)
B = np.zeros(shape=(2, 2))
# B = np.expand_dims(B, axis=0)



# A = np.concatenate([A, B], axis=0)
# C = np.concatenate([A, B], axis=0)
print(C)
print(C.shape)

path = os.getcwd()
new_dir = 'test3_dir'
filename = os.path.join(path, new_dir)
try: 
    os.mkdir(filename) 
except OSError as error: 
    print(error)  