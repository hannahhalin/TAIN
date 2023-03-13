## adapted from https://github.com/zacjiang/GMA/blob/main/core/gma.py
import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

                
class Attention(nn.Module):
    def __init__(
        self,
        dim1,
        dim2,
        dim_head = 128
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.vtNorm = nn.InstanceNorm2d(dim_head)
        self.to_k = nn.Conv2d(dim1, dim_head, 1, bias=False)
        self.to_q = nn.Conv2d(dim2, dim_head, 1, bias=False)
        self.to_qk = nn.Conv2d(dim1, dim_head, 1, bias=False)

        self.to_k2 = nn.Sequential(nn.Conv2d(dim1, dim_head, 1),
                                  nn.ReLU(True),
                                  nn.Conv2d(dim_head, dim_head, 1),
                                  nn.ReLU(True),
                                  nn.Conv2d(dim_head, dim1, 1),
                                  )
        self.to_q2 = nn.Sequential(nn.Conv2d(dim2, dim_head, 1),
                                  nn.ReLU(True),
                                  nn.Conv2d(dim_head, dim_head, 1),
                                  nn.ReLU(True),
                                  nn.Conv2d(dim_head, dim2, 1),
                                  )
        
    def forward(self, fmap1, fmap2, dmap):
        b, c, h, w = fmap1.shape

        k1 = self.to_qk(fmap1)
        k2 = self.to_qk(fmap2)
        q = self.to_qk(dmap)
        
                
        k1 = F.normalize(k1, dim=1)
        k2 = F.normalize(k2, dim=1)
        q = F.normalize(q, dim=1)

        q, k1, k2 = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=1), (q, k1, k2))

        sim1 = einsum('b h x y d, b h u v d -> b h x y u v', q, k1)
        sim2 = einsum('b h x y d, b h u v d -> b h x y u v', q, k2)
                        
        sim1 = rearrange(sim1, 'b h x y u v -> b h (x y) (u v)')
        sim2 = rearrange(sim2, 'b h x y u v -> b h (x y) (u v)')
                                
        attn1 = sim1.softmax(dim=-1)
        attn2 = sim2.softmax(dim=-1)
        
        return attn1, attn2 #, attn2


class Aggregate(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 128
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.to_v = nn.Conv2d(dim, dim_head, 1, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1))

        if dim != dim_head:
            self.project = nn.Conv2d(dim_head, dim, 1, bias=False)
        else:
            self.project = None

        
    def forward(self, attn, fmap):
        b, c, h, w = fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=1)
                              
        attnMax, attnMaxI = torch.max(attn, dim=3)
        attnMax = rearrange(attnMax, 'b h (x y) -> b h x y',  h=1, x=h, y=w)
        
        out = self.batched_index_select(v,2,attnMaxI)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=1, x=h, y=w)
                              
        if self.project is not None:
            out = self.project(out)
             
        out = fmap + self.alpha * out
            
        return out, attnMax
        
    def batched_index_select(self, input, dim, index):
        #  adapted from https://github.com/jinfagang/yolov7_d2/blob/main/tests.py
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
        
