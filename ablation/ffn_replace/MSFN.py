import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import *
from einops import rearrange


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W],device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x




class Attention(nn.Module):
    def __init__(self, channel, win_size, num_heads, bias):
        super(Attention, self).__init__()

        self.channel = channel
        self.win_size = win_size
        self.num_heads = num_heads
        self.head_dim = channel // num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.ln1 = LayerNorm2d(channel)
        self.conv1  = nn.Conv2d(channel, channel*4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channel, channel*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias)
       


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.ln1(x)
        x_win, win_nums = window_partitionx(x, self.win_size) # (win_num*b), c, win_h, win_w
       
        qQKV = self.conv1(x_win)
        q,Q,K,V = torch.chunk(qQKV, chunks=4, dim=1)
        
        Q = self.maxpool(Q)  
        K = self.maxpool(K)
        V = self.maxpool(V)

        Q = rearrange(Q, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        K = rearrange(K, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        V = rearrange(V, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        attn = (Q @ K.transpose(-2, -1)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = attn @ V
        out = rearrange(out, 'b n (h w) c -> b (n c) h w',h=self.win_size//2)   

        kv = self.conv2(out)
        k,v = torch.chunk(kv,chunks=2, dim=1)
        q = rearrange(q, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        k = rearrange(k, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        v = rearrange(v, 'b (n c) h w -> b n (h w) c', n=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b n (h w) c -> b (n c) h w',h=self.win_size) 
        out = self.conv3(out)

        out = window_reversex(out, self.win_size, h, w, win_nums) # b, c, h, w

        return out
    
    def flops(self, H, W):

        flops = 0
        B = (H / self.win_size) * (W / self.win_size)

        flops += B * self.channel * 256 * 256    # Q @ K^T     
        flops += B * 256 * 256 * self.channel   # attn @ V
        flops += B * self.channel * 1024 * 256  # q @ k^T
        flops += B * 256 * 256 * self.channel   # attn @ v
        
        return flops


##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x   



##########################################################################
# block
class Block(nn.Module):
    def __init__(self, channel, win_size, num_heads, factor, bias=False):
        super(Block, self).__init__()

        self.ln = LayerNorm2d(channel)

        self.attention = Attention(channel=channel, win_size=win_size, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim=channel,ffn_expansion_factor=factor, bias=bias)

    def forward(self, x):

        x = self.attention(x) + x
        x = self.ffn(self.ln(x)) + x
        return x 

    def flops(self, H, W):
        
        flops = 0
        
        flops += self.attention.flops(H, W)  # attention~~@
        
        return flops


##########################################################################
# layer
class Layer(nn.Module):
    def __init__(self, channel, win_size, num_heads,factor, depth, bias=False):
        super(Layer, self).__init__()
        self.layers = nn.ModuleList([
            Block(channel=channel, win_size=win_size, num_heads=num_heads,factor=factor, bias=bias) for _ in range(depth)
        ])

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x

    def flops(self, H, W):

        flops = 0
        for block in self.layers:
            flops += block.flops(H, W)

        return flops


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, channel, bias):
        super(OverlapPatchEmbed, self).__init__()
        self.in_c = in_c
        self.channel = channel
        self.proj = nn.Conv2d(in_c, channel, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act_layer = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.proj(x)
        x = self.act_layer(x)

        return x



##########################################################################
## Resizing modules
class DownSample(nn.Module):
    def __init__(self, channel, bias):
        super(DownSample, self).__init__()
        self.channel = channel
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=3, stride=2, padding=1, bias=bias), 
            nn.ReLU(True)
        )

    def forward(self, x):
        
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channel, bias):
        super(UpSample, self).__init__()
        self.channel = channel
        self.body = nn.Sequential(
            nn.Conv2d(channel*2, channel*4, kernel_size=1, bias=bias),
            nn.PixelShuffle(2), 
        )

    def forward(self, x):

        return self.body(x)



class Net(nn.Module):
    def __init__(self, 
                inp_channels = 3, 
                out_channels = 3, 
                channel = 32, 
                win_size = [64,64,64], 
                heads = [1,2,4], 
                factor = 2.66,
                depth = [4,4,16],
                bias = False
        ):

        super(Net, self).__init__()

        self.channel = channel

        self.patch_embed = OverlapPatchEmbed(inp_channels, channel, bias=bias)

        ##### Detail Stage
        self.Layers = nn.ModuleList([
            Layer(channel=channel,   win_size=win_size[0], num_heads=heads[0],factor=factor, depth=depth[0], bias=bias),
            Layer(channel=channel*2, win_size=win_size[1], num_heads=heads[1],factor=factor, depth=depth[1], bias=bias),
            Layer(channel=channel*4, win_size=win_size[2], num_heads=heads[2],factor=factor, depth=depth[2], bias=bias),
            Layer(channel=channel*4, win_size=win_size[2], num_heads=heads[2],factor=factor, depth=depth[2], bias=bias),
            Layer(channel=channel*2, win_size=win_size[1], num_heads=heads[1],factor=factor, depth=depth[1], bias=bias),
            Layer(channel=channel,   win_size=win_size[0], num_heads=heads[0],factor=factor, depth=depth[0], bias=bias),
        ])

        self.Down = nn.ModuleList([
            DownSample(channel=channel,   bias=bias), 
            DownSample(channel=channel*2, bias=bias), 
        ])

        self.Up = nn.ModuleList([
            UpSample(channel=channel,   bias=bias), 
            UpSample(channel=channel*2, bias=bias), 
        ])

        self.output = nn.Conv2d(channel, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):

        z = self.patch_embed(inp_img)

        ### Encoder 1
        res1 = self.Layers[0](z)
        z = self.Down[0](res1)

        ### Encoder 2
        res2 = self.Layers[1](z)
        z = self.Down[1](res2)

        ### bottleneck
        z = self.Layers[2](z)
        z = self.Layers[3](z)

        ### Decoder 2
        z = self.Up[1](z)
        z = self.Layers[4](z + res2)

        ### Decoder 1
        z = self.Up[0](z)
        z = self.Layers[5](z + res1)
    
        out = self.output(z)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        return out + inp_img

    def flops(self, x):
        _, _, H, W = x.shape

        flops = 0
        flops += self.Layers[0].flops(H, W)
        flops += self.Layers[1].flops(H / 2, W/ 2)
        flops += self.Layers[2].flops(H / 4, W/ 4)
        flops += self.Layers[3].flops(H / 4, W / 4)
        flops += self.Layers[4].flops(H / 2, W / 2)
        flops += self.Layers[5].flops(H, W)
        
        return flops
