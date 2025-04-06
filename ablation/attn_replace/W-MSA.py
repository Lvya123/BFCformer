import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import *
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


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
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v    
    
    def flops(self, q_L, kv_L=None): 
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v


#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!") 
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'



########### self-attention #############
class Attention(nn.Module):
    def __init__(self, dim,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
            
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
    
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'




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
    def __init__(self, channel, win_size=8, num_heads=1, shift_size =0, qkv_bias=True, qk_scale=None,drop=0., attn_drop=0.,token_projection='linear',bias=False):
        super(Block, self).__init__()

        self.ln = LayerNorm2d(channel)   
        self.win_size = 8
        
        self.attention = WindowAttention(
            channel, win_size=to_2tuple(win_size), num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)
        self.ffn = FFN(channel=channel, bias=bias)

    def forward(self, x):

        x1 = x
        x = self.ln(x)  # B C H W

        # cyclic shift
        # if shift_size > 0:
        #     x2 = rearrange(x,'b c h w -> b h w c')
        #     shifted_x = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        #     shifted_x = rearrange(shifted_x,'b h w c -> b c h w')
        # else:
        #     shifted_x = x

        x, win_nums = window_partitionx(x, self.win_size)  # B' ws ws C
        x = rearrange(x,'b c h w -> b (h w) c')
        x = self.attention(x) 
        x = rearrange(x,'b (h w) c -> b c h w',h=self.win_size)
        x = window_reversex(x, self.win_size, x1.shape[2], x1.shape[3], win_nums)   # B C H W

        # # reverse cyclic shift
        # if shift_size > 0:
        #     shifted_x = rearrange(x,'b c h w -> b h w c')
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        #     x = rearrange(shifted_x,'b h w c -> b c h w')
        # else:
        #     x = x

        x = x + x1

        x = self.ffn(x) + x
        return x 



##########################################################################
# layer
class Layer(nn.Module):
    def __init__(self, channel, win_size=8, heads=1, shift_size=0, qkv_bias=True, qk_scale=None,drop=0., attn_drop=0.,token_projection='linear', depth=[4,4,8], bias=False):
    
        super(Layer, self).__init__()
        self.layers = nn.ModuleList([
            Block(channel=channel, win_size=win_size, num_heads=heads, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,drop=drop, attn_drop=attn_drop,token_projection=token_projection, bias=bias) for _ in range(depth)
        ])

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x


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
                heads = [1,2,4], 
                depth = [4,4,8],
                bias = False
        ):

        super(Net, self).__init__()

        self.channel = channel

        self.patch_embed = OverlapPatchEmbed(inp_channels, channel, bias=bias)

        ##### Detail Stage
        self.Layers = nn.ModuleList([
            Layer(channel=channel,   win_size=8, heads = heads[0], depth=depth[0], bias=bias),
            Layer(channel=channel*2, win_size=8, heads = heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel*4, win_size=8, heads = heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*4, win_size=8, heads = heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*2, win_size=8, heads = heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel,   win_size=8, heads = heads[0], depth=depth[0], bias=bias),
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
