import time
import torch

a = torch.randn(2560, 500, 500).cuda()
c = torch.randn(256, 500, 500)
for i in range(200):
    # d = torch.einsum("bmn,bnk->bmk", c, c)
    b = torch.einsum("bmn,bnk->bmk", a, a)
    time.sleep(0.0001)
    