import time
import torch

a = torch.randn(256, 5000, 5000)
for i in range(5):
    c = torch.randn(256, 5000, 5000)
    b = torch.einsum("bmn,bnk->bmk", a, a)
    time.sleep(0.0001)
    