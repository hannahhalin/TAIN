import torch
import torch.nn as nn
from einops import rearrange
from .vt import Attention, Aggregate


class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_feat, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out


# Channel Attention (CA) Module
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, out_channel=None, reshapeI=False):
        super(CALayer, self).__init__()
        
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reshapeI = reshapeI
        if out_channel is None:
            out_channel = channel
            
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, out_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, x2=None):
        if x2 is not None:
            y = torch.cat([x, x2], dim=1)
        else:
            y = self.avg_pool(x)
        y = self.conv_du(y)
        
        if self.reshapeI:
            x[:,:x.size(1)//2,:,:] *= y[:,:1,:,:].tile((1,x.size(1)//2,1,1))
            x[:,x.size(1)//2:,:,:] *= y[:,1:,:,:].tile((1,x.size(1)//2,1,1))
            return x, y
        else:
            return x * y, y


# Image Attention (IA) Module
class IALayer(nn.Module):
    def __init__(self, channel, reduction=16, out_channel=None, reshapeI=False):
        super(IALayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reshapeI = reshapeI
        if out_channel is None:
            out_channel = channel
            
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, out_channel, 1, padding=0, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x, x2=None):
        if x2 is not None:
            y = torch.cat([x, x2], dim=1)
        else:
            y = self.avg_pool(x)
        y = self.conv_du(y)
        
        if self.reshapeI:
            x[:,:x.size(1)//2,:,:] *= y[:,:1,:,:].tile((1,x.size(1)//2,1,1))
            x[:,x.size(1)//2:,:,:] *= y[:,1:,:,:].tile((1,x.size(1)//2,1,1))
            
            return x, y
        else:
            return x * y, y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, bias=True,
            norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()
            
        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=2 if downscale else 1, norm=norm),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1, norm=norm),
            CALayer(out_feat, reduction)
        )
            
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca
 
    def forward(self, x):
        if type(x) is list:
            res = x[0]
            x = torch.cat(x, dim=1)
        else:
            res = x
            
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, Block, n_resblocks, n_feat, kernel_size, reduction, act, norm=False, att_scI=False, n_resgroups=5):
        super(ResidualGroup, self).__init__()
        self.att_scI =att_scI
        self.IALayerI = True
        self.n_resgroups = n_resgroups
        
        # visual transformer
        self.att = Attention(dim1=n_feat, dim2=n_feat, dim_head=n_feat)
        self.agg = Aggregate(dim=n_feat, dim_head=n_feat)
        
        modules_body = [Block(n_feat*3, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act)]
        
        for _ in range(n_resblocks-1):
            modules_body.append(Block(n_feat, n_feat, kernel_size, reduction, bias=True, norm=norm, act=act))
        modules_body.append(ConvNorm(n_feat, n_feat, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.IALayer = IALayer((n_feat+1)*2, out_channel=2, reshapeI=True)
                    
                    
    def forward(self, x, x0=None, x1=None, xout=None):
    
        # CS transformer module
        if xout is not None:
            attention0, attention1 = self.att(fmap1=x0, fmap2=x1, dmap=xout)
        else:
            attention0, attention1 = self.att(fmap1=x0, fmap2=x1, dmap=x)
  
        # aggregate features
        vid_features_global0, attMax0 = self.agg(attention0, x0)
        vid_features_global1, attMax1 = self.agg(attention1, x1)
            
                        
        attMax = torch.cat([attMax0, attMax1], dim=1) # b 2 1 xy
        vid_features_global = torch.stack([vid_features_global0, vid_features_global1], dim=1) # b 2 d x y
        vid_features_global = rearrange(vid_features_global, 'b c d x y -> b (c d) x y') # b 2d x y
        if self.IALayerI:
            vid_features_global, vid_features_global_att = self.IALayer(vid_features_global, attMax)
        xs = [x, vid_features_global]
            
        res = self.body(xs)
        res += x
                     
        return res


def conv(in_channels, out_channels, kernel_size, 
         stride=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        stride=1,
        bias=bias,
        groups=groups)


def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class Interpolation(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_feats, reduction=16, 
                 act=nn.LeakyReLU(0.2, True), norm=False):
        super(Interpolation, self).__init__()
        
        # define modules: head, body, tail
        self.headConv = conv3x3(n_feats*2, n_feats)

        modules_body = [
                ResidualGroup(
                    RCAB,
                    n_resblocks=n_resblocks,
                    n_feat=n_feats,
                    kernel_size=3,
                    reduction=reduction,
                    act=act,
                    norm=norm,
                    n_resgroups=n_resgroups)
                for _ in range(n_resgroups)]
            
        self.body = nn.Sequential(*modules_body)
        self.tailConv = conv3x3(n_feats, n_feats)
            
    def forward(self, x0, x1):
        
        # build input tensor
        x = torch.cat([x0, x1], dim=1)
            
        x = self.headConv(x)
        
        xout = None
        for m in self.body:
            x = m(x, x0=x0, x1=x1,xout=xout)
                    
            xout = self.tailConv(x)
                    
        out = self.tailConv(x)
        
        return out
