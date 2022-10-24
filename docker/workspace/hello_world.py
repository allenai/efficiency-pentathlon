import time
import torch

a = torch.randn(256, 500, 500).cuda()
for i in range(50):
    c = torch.randn(256, 500, 500)
    d = torch.einsum("bmn,bnk->bmk", c, c)
    b = torch.einsum("bmn,bnk->bmk", a, a)
    time.sleep(0.0001)
    