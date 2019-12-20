import torch

def to_tsm(x,batch_size, _, d, h, w):
    x = x.transpose(1,2).view(batch_size*d, _, h, w).contiguous()
    return x

def to_seg(x,d):
    Nd, c, h,w =x.size()
    return x.view(Nd//d,d,c,h,w).transpose(1,2).contiguous()
    #slices = torch.chunk(x, d, dim=0)
   # recons = torch.cat(slices[0:d],dim=1)
   # return recons