
import torch
import torch.nn as nn
from .common import Interpolation


class TAIN(nn.Module):
    def __init__(self, depth=3, n_resgroups=5,in_channels=3):
        super(TAIN, self).__init__()
        """
            TAIN for video interpolation
            
            Args:
                depth - number of down-sampling in PixelShuffle
                n_resgroups - number of resgroups
                in_channels - number of input channels
        """
        
        # down-shuffle : expand in channel dimension
        self.shuffler = PixelShuffle(1 / 2**depth)
        
        # interpolate intermediate frame
        self.interpolate = Interpolation(n_resgroups, 12, in_channels * (4**depth))
        
        # up-shuffle : expand in spatial dimension
        self.shuffler_out = PixelShuffle(2**depth)
        

    def forward(self, x1, x2):
        
        x1, x2, mi = sub_mean(x1,x2)
        
        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)

        
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)
        
        feats = self.interpolate(feats1, feats2)
                
        out = self.shuffler_out(feats)
        
        if not self.training:
            out = paddingOutput(out)
                         
        out += mi
                
        return out


def sub_mean(x1, x2):
    mean1 = x1.mean(2, keepdim=True).mean(3, keepdim=True)
    mean2 = x2.mean(2, keepdim=True).mean(3, keepdim=True)
    mean = (mean1 + mean2) / 2
    x1 -= mean
    x2 -= mean
    return x1, x2, mean
    

def InOutPaddings(x):
    w, h = x.size(3), x.size(2)
    padding_width, padding_height = 0, 0
    if w != ((w >> 7) << 7):
        padding_width = (((w >> 7) + 1) << 7) - w
    if h != ((h >> 7) << 7):
        padding_height = (((h >> 7) + 1) << 7) - h
        
    paddingInput = nn.ReflectionPad2d(padding=[padding_width // 2, padding_width - padding_width // 2,
                                               padding_height // 2, padding_height - padding_height // 2])
    paddingOutput = nn.ReflectionPad2d(padding=[0 - padding_width // 2, padding_width // 2 - padding_width,
                                                0 - padding_height // 2, padding_height // 2 - padding_height])
                         
    return paddingInput, paddingOutput

    
def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)
