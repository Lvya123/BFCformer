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



##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    


class FFN(nn.Module):
    def __init__(self, channel, bias):
        super().__init__()
        self.channel = channel
        self.dim = channel//4
        self.dim_untouched = channel - self.dim

        self.ln = LayerNorm2d(channel)
        self.PConv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=bias)
        self.conv0 = nn.Conv2d(channel, channel*4, 1, 1, 0, bias=bias)
        self.conv1 = nn.Conv2d(channel, channel*4, 1, 1, 0, bias=bias)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv2d(channel*4, channel, 1, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(channel*4, channel*4, 3, 1, 1, groups=channel*4, bias=bias)
        self.gelu3 = nn.GELU()
        self.conv4 = nn.Conv2d(channel*4, channel, 1, 1, 0, bias=bias)

        self.patch_size = 8
        self.fft_weight1 = nn.Parameter(torch.ones((channel, self.patch_size, self.patch_size // 2 + 1))) 
        self.fft_weight2 = nn.Parameter(torch.ones((channel, self.patch_size, self.patch_size // 2 + 1))) 
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.ln(x)
        x1 = x
        y1,y2 = torch.split(x, [self.dim,self.dim_untouched], dim=1)
        y1 = self.PConv(y1)
        x2 = torch.cat((y1,y2),dim=1)
        
        x1 = self.conv0(x1)
        x1 = self.gelu1(x1)
        x1 = self.conv2(x1)
        x1, win_nums1 = window_partitionx(x1, self.patch_size)
        x1 = torch.fft.rfft2(x1)
        x1 = self.fft_weight1 * x1 
        x1 = torch.fft.irfft2(x1, s=(self.patch_size, self.patch_size))
        x1 = window_reversex(x1, self.patch_size, h, w, win_nums1)
        
        x2 = self.conv1(x2)
        x2 = self.gelu2(x2)
        x2 = self.conv3(x2)
        x2 = self.gelu3(x2)
        x2 = self.conv4(x2)
        x2, win_nums2 = window_partitionx(x2, self.patch_size)
        x2 = torch.fft.rfft2(x2)
        x2 = self.fft_weight2 * x2 
        x2= torch.fft.irfft2(x2, s=(self.patch_size, self.patch_size))
        x2 = window_reversex(x2, self.patch_size, h, w, win_nums2)
        
        out = x1 + x2

        return out

##########################################################################
# block
class Block(nn.Module):
    def __init__(self, channel, num_heads, bias=False):
        super(Block, self).__init__()

        self.ln = LayerNorm2d(channel)

        self.attention = Attention(channel, num_heads, bias)
        self.ffn = FFN(channel=channel, bias=bias)

    def forward(self, x):

        x = self.attention(self.ln(x)) + x
        x = self.ffn(x) + x
        return x 

    # def flops(self, H, W):
        
    #     flops = 0
        
    #     flops += self.attention.flops(H, W)  # attention~~@
        
    #     return flops


##########################################################################
# layer
class Layer(nn.Module):
    def __init__(self, channel, num_heads, depth, bias=False):
        super(Layer, self).__init__()
        self.layers = nn.ModuleList([
            Block(channel=channel, num_heads=num_heads, bias=bias) for _ in range(depth)
        ])

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x

    # def flops(self, H, W):

    #     flops = 0
    #     for block in self.layers:
    #         flops += block.flops(H, W)

    #     return flops


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
                depth = [4,4,16],
                bias = False
        ):

        super(Net, self).__init__()

        self.channel = channel

        self.patch_embed = OverlapPatchEmbed(inp_channels, channel, bias=bias)

        ##### Detail Stage
        self.Layers = nn.ModuleList([
            Layer(channel=channel,   num_heads=heads[0], depth=depth[0], bias=bias),
            Layer(channel=channel*2, num_heads=heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel*4, num_heads=heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*4, num_heads=heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*2, num_heads=heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel,   num_heads=heads[0], depth=depth[0], bias=bias),
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

    # def flops(self, x):
    #     _, _, H, W = x.shape

    #     flops = 0
    #     flops += self.Layers[0].flops(H, W)
    #     flops += self.Layers[1].flops(H / 2, W/ 2)
    #     flops += self.Layers[2].flops(H / 4, W/ 4)
    #     flops += self.Layers[3].flops(H / 4, W / 4)
    #     flops += self.Layers[4].flops(H / 2, W / 2)
    #     flops += self.Layers[5].flops(H, W)
        
    #     return flops
