# test sinkhorn loss
import torch
from geomloss import SamplesLoss
from time import time

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

N, M = (5000, 5000)
x, y = torch.randn(N, 3).type(dtype), torch.randn(M, 3).type(dtype)
x, y = x / (2 * x.norm(dim=1, keepdim=True)), y / (2 * y.norm(dim=1, keepdim=True))
x.requires_grad = True

Loss = SamplesLoss(
    'sinkhorn', blur=0.05, backend='auto'
)
t_0 = time()
L_xy = Loss(x, y)
L_xy.backward()
t_1 = time()
print("{:.2f}s, cost = {:.6f}".format(t_1 - t_0, L_xy.item()))
