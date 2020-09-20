import torch
import torch.nn as nn
import torch.nn.functional as F

def test(a,b,c, d=4, e=5):
    print(a+b+c)
    print(d+e)


lol = {'d':4, 'e':5}
test(1,2,3, **lol)
