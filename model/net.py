import warnings
from functools import partial
import sys
import os
import thop
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from sam2.build_sam import build_sam2
from einops import rearrange
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention
import numbers

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x, timesteps, cond_img):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0])) #先嵌入，在通过time_embed映射，Bxembed_dims[0]
        time_token = time_token.unsqueeze(dim=1) # Bx1xembed_dims[0]

        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0] #提取时间步嵌入的位置（第一个位置）,[B, embed_dim(0)]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[B, embed_dim, H, W]
        outs.append(x)

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1) #[B, 1, embed_dims[1]]
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, timesteps, cond_img):
        x = self.forward_features(x, timesteps, cond_img)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# [B, H*W, embed_dim]
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# [B, H*W, embed_dim]
class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# 下采样2倍 输出通道数x4
def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )

# 上采样两倍 输出通道数不变
class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.timeattn1 = TimeCrossAttn(cond_dim=144, time_dim=144)
        self.timeattn2 = TimeCrossAttn(cond_dim=288, time_dim=288)
        self.timeattn3 = TimeCrossAttn(cond_dim=576, time_dim=576)
        self.timeattn4 = TimeCrossAttn(cond_dim=1152, time_dim=1152)
        self.timelinear1 = nn.Linear(in_features=144, out_features=288)
        self.timelinear2 = nn.Linear(in_features=288, out_features=576)       
        self.timelinear3 = nn.Linear(in_features=576, out_features=1152)



        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim
        
        self.conv_c1 = nn.Conv2d(144, 1, kernel_size=1)
        self.conv_c2 = nn.Conv2d(288, 1, kernel_size=1) 

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.final_conv = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim * 2, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim * 8, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        
        self.fft = BlockFFT(dim=1152, h=11, w=11)

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        #下采样4倍，同时由resnet引入t
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        #新增，下采样2倍，同时由resnet引入t
        self.down_2x_1 = nn.Sequential(
            ConvModule(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(embedding_dim, embedding_dim*2, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )

        self.down_2x_2 = nn.Sequential(
            ConvModule(embedding_dim*2, embedding_dim*4, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),
            ConvModule(embedding_dim*4, embedding_dim*4, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )


        self.down_2x_3 = nn.Sequential(
            ConvModule(embedding_dim*4, embedding_dim*8, kernel_size=3, stride=2, padding=1, norm_cfg=dict(type='BN')),  # 3x3卷积+步长2 -> 下采样
            ConvModule(embedding_dim*8, embedding_dim*8, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))  # 1x1卷积调整通道数
        )
        self.cat_conv = ConvModule(in_channels=embedding_dim * 16, out_channels=embedding_dim*8, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        # 上采样4倍 通道数变为1/8
        self.up_2x_3 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*8, out_channels=embedding_dim*4, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*4, embedding_dim*4, factor=2),
            ConvModule(in_channels=embedding_dim*4, out_channels=embedding_dim*4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*4, out_channels=embedding_dim*2, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*2, embedding_dim*2, factor=2),
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim*2, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_1 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim, embedding_dim, factor=2),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            # ConvModule(in_channels=embedding_dim//8+1, out_channels=embedding_dim//8, kernel_size=1,
            #            norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )
        self.cond_attention1 = MultiScaleSpatialAttention(144)
        self.cond_attention2 = MultiScaleSpatialAttention(288)
        self.cond_attention3 = MultiScaleSpatialAttention(576)
        self.cond_fusion = ProgressiveFusion()
        self.feature_fusion = feature_fusion(channels=1152)
    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))#8*144
        t1 = self.timelinear1(t)
        t2 = self.timelinear2(t1)
        t3 = self.timelinear3(t2)
        c1, c2, c3, c4 = inputs  # c1:144*88*88 c2:288*44*44...
        ##############################################
        c1 = self.timeattn1(c1, t)
        c2 = self.timeattn2(c2, t1)
        c3 = self.timeattn3(c3, t2)
        c4 = self.timeattn4(c4, t3)
        _x = [x]
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c1_attention = self.cond_attention1(c1)
        ############## MLP decoder on C1-C4 ###########
        x = x + x*c1_attention

        for blk in self.down_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c2_attention = self.cond_attention2(c2)
        x = x + x*c2_attention     

        for blk in self.down_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        c3_attention = self.cond_attention3(c3)
        x = x + x*c3_attention
        for blk in self.down_2x_3:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x) # 1152*11*11

        c_final = self.cond_fusion(c1, c2, c3, c4)
        

        #傅里叶变换
        x = self.fft(x)
        # fusion x_feat and x then transposed conv   貌似将融合后的条件又与xt融合了
        x = self.feature_fusion(x, c_final, t)
        
        for blk in self.up_2x_3:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        # x = self.pred(torch.cat([x, _x.pop(-1)], dim=1))
        x = self.pred(x)
        return x


class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        checkpoint_path = '/media/user/HDD/ssx_project/first_paper/CamoDiffusion_two/pretrained_weights/sam2_hiera_large.pt'
        self.backbone = SAM2backbone(checkpoint_path=checkpoint_path)
        self.depth_backbone = SAM2backbone_depth(checkpoint_path=checkpoint_path)
        self.decode_head = Decoder(dims=[144, 288, 576, 1152], dim=144, class_num=class_num, mask_chans=mask_chans)
       #self._init_weights()  # load pretrain

        self.frequency_fusion_stages =  nn.ModuleList([
            # Level 1: 144x88x88
            LightCrossAttention(in_dim_rgb=144, in_dim_depth=144, embed_dim=144),
            
            # Level 2: 288x44x44
            LightCrossAttention(in_dim_rgb=288, in_dim_depth=288, embed_dim=288),
            
            # Level 3: 576x22x22
            LightCrossAttention(in_dim_rgb=576, in_dim_depth=576, embed_dim=576),
            
            # Level 4: 1152x11x11
            LightCrossAttention(in_dim_rgb=1152, in_dim_depth=1152, embed_dim=1152)
        ])


    def forward(self, x, timesteps, cond_img, depth_map):

        rgb_features = self.backbone(x, timesteps, cond_img)
        depth_features = self.depth_backbone(x, timesteps ,depth_map)
        features = []
        # print("timestep:",timesteps.shape)

        for i, (rgb_feat, depth_feat) in enumerate(zip(rgb_features, depth_features)):
            # 执行特征融合
            fused = self.frequency_fusion_stages[i](rgb_feat, depth_feat)
           
            features.append(fused)
        features = self.decode_head(features, timesteps, x)
        return features

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load('/media/user/HDD/ssx_project/first_paper/CamoDiffusion_two/pretrained_weights/sam2_hiera_large.pt') #for save mem
        model_dict = self.backbone.state_dict()
        model_dict_depth = self.depth_backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict_depth = {k: v for k, v in pretrained_dict.items() if k in model_dict_depth}
        model_dict.update(pretrained_dict)
        model_dict_depth.update(pretrained_dict_depth)
        self.backbone.load_state_dict(model_dict, strict=False)
        self.depth_backbone.load_state_dict(model_dict_depth, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img, depth_map):
        return self.forward(x, timesteps, cond_img, depth_map)

    def extract_features(self, cond_img, depth_map):
        # do nothing
        return cond_img, depth_map


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass

#SAM2部分
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
class SAM2backbone(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2backbone, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.proj = nn.Conv2d(3, 3, kernel_size=3, stride=1,
                              padding=(1, 1))
        self.x_proj = nn.Conv2d(1, 3, kernel_size=3, stride=1,
                                       padding=(1, 1))
            # set mask_proj weight to 0
        self.x_proj.weight.data.zero_()
        self.x_proj.bias.data.zero_()
        self.time_embed = nn.Sequential(
            nn.Linear(144, 4 * 144),
            nn.SiLU(),
            nn.Linear(4 * 144, 144),
        )
        self.time_processor = nn.Sequential(
            nn.Linear(144, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 16 * 16),  # 展开为16x16特征图
            nn.Unflatten(1, (256, 16, 16)),  # [B, 256, 16, 16]
            
            # 特征解码器
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),  # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),   # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),    # 64->128
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),    # 128->256
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),                       # 保持256->256
            nn.Upsample(size=(352, 352), mode='bilinear'),       # 调整到目标尺寸
            nn.Sigmoid()  # 最终门控图 [B, 1, 352, 352]
        )
    def forward(self, x, timesteps, cond_img):
        #后续考虑每个stage加上时间步嵌入
        t = self.time_embed(timestep_embedding(timesteps, 144))#8*144
        t = self.time_processor(t)
        cond_img = cond_img + x*t
        x1, x2, x3, x4 = self.encoder(cond_img)
        return x1, x2, x3, x4
    
class SAM2backbone_depth(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2backbone_depth, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.weight_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()  # 输出权重图 [0,1]
        )
        self.time_embed = nn.Sequential(
            nn.Linear(144, 4 * 144),
            nn.SiLU(),
            nn.Linear(4 * 144, 144),
        )
        self.time_processor = nn.Sequential(
            nn.Linear(144, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 16 * 16),  # 展开为16x16特征图
            nn.Unflatten(1, (256, 16, 16)),  # [B, 256, 16, 16]
            
            # 特征解码器
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),  # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),   # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),    # 64->128
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),    # 128->256
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),                       # 保持256->256
            nn.Upsample(size=(352, 352), mode='bilinear'),       # 调整到目标尺寸
            nn.Sigmoid()  # 最终门控图 [B, 1, 352, 352]
        )
    def forward(self, x, timesteps, cond_img):
        #后续考虑每个stage加上时间步嵌入
        t = self.time_embed(timestep_embedding(timesteps, 144))#8*144
        t =self.time_processor(t)
        cond_img = cond_img + t * x 
        cond_img = torch.cat([cond_img, cond_img, cond_img], dim=1)

        x1, x2, x3, x4 = self.encoder(cond_img)
        return x1, x2, x3, x4
    
class BlockFFT(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w//2 + 1, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')                # 转换为频域
        x = x * torch.view_as_complex(self.complex_weight)               # 频域滤波
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')      # 转换回空域
        return x.reshape(B, C, H, W)

class SAM_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Step 1: 通道压缩
        self.conv1 = nn.Conv2d(144, 64, kernel_size=1)
        
        # Step 2: 第一次上采样
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Step 3: 特征细化
        self.refine1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Step 4: 第二次上采样
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Step 5: 预测头
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)      # [8, 64, 88, 88]
        x = self.up1(x)        # [8, 64, 176, 176]
        x = self.refine1(x)    # [8, 64, 176, 176]
        x = self.up2(x)        # [8, 32, 352, 352]
        x = self.head(x)       # [8, 1, 352, 352]
        return x

class TimeCrossAttn(nn.Module):
    def __init__(self, cond_dim, time_dim=144, num_heads=12, residual=True):
        super().__init__()
        assert time_dim % num_heads == 0, "time_dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = time_dim // num_heads  # 144/12=12
        self.residual = residual

        # 条件特征投影（无需时间嵌入投影）
        self.key = nn.Conv2d(cond_dim, time_dim, kernel_size=1)
        self.value = nn.Conv2d(cond_dim, time_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(time_dim, cond_dim, kernel_size=1)

    def forward(self, cond, t):
        B, C, H, W = cond.shape
        residual = cond
        
        # 直接使用t作为Query [B, 144] → [B, num_heads, head_dim]
        q = t.view(B, self.num_heads, self.head_dim).unsqueeze(-1)  # [B, H, D, 1]
        
        # 投影条件特征
        k = self.key(cond).view(B, self.num_heads, self.head_dim, H*W)
        v = self.value(cond).view(B, self.num_heads, self.head_dim, H*W)
        
        # 注意力计算
        scale = self.head_dim ** 0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)  # [B, H, D, 1]
        
        # 输出处理
        out = out.view(B, -1, 1, 1)
        out = self.out_proj(out).expand_as(cond)
        
        return out + residual if self.residual else out




# spatial rgb-d fusion
class EnhancedChannelAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        # 自适应池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出 [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化，输出 [B, C, 1, 1]
        
        # 压缩-激励结构
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),  # 激励
            nn.Sigmoid()  # 输出通道权重 [0,1]
        )

    def forward(self, x):
        """
        输入: 
            x : 特征图 [B, C, H, W]
        输出: 
            weights : 通道权重 [B, C, 1, 1]
        """
        b, c, _, _ = x.size()
        
        # 平均池化分支
        avg_out = self.avg_pool(x)  # [B, C, 1, 1]
        avg_out = avg_out.view(b, c)  # [B, C]
        
        # 最大池化分支
        max_out = self.max_pool(x)  # [B, C, 1, 1]
        max_out = max_out.view(b, c)  # [B, C]
        
        # 融合双路池化结果
        combined = avg_out + max_out  # [B, C]
        
        # 生成通道权重
        channel_weights = self.fc(combined)  # [B, C]
        channel_weights = channel_weights.view(b, c, 1, 1)  # [B, C, 1, 1]
        
        return channel_weights

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernels=[1,3,5]):
        super().__init__()
        
        # 三个并行分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                # 尺度相关的卷积层
                nn.Conv2d(in_channels, in_channels, k, padding=k//2, bias=False),
                # 空间注意力模块
                SpatialAttention()
            ) for k in kernels
        ])
        
    def forward(self, x):
        # 各分支独立处理
        branch_outs = [branch(x) for branch in self.branches]
        
        # 权重图相加融合
        summed_attn = torch.stack(branch_outs).sum(dim=0)
        return summed_attn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 空间注意力生成器
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道维度聚合
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        
        # 生成空间注意力图
        attn = self.spatial_attn(combined)  # [B,1,H,W]
        return attn



class feature_fusion(nn.Module):
    def __init__(self, channels=1152, t_dim=144):
        super().__init__()
        # 时间嵌入处理
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, 11*11),
            nn.GELU(),
            nn.Unflatten(1, (1, 11, 11)),
            nn.Sigmoid()
        )
        # 协同通道注意力
        self.channel_attn = EnhancedChannelAttention(channels)
        
        # 共享归一化层
        self.norm = nn.Sequential(
            nn.LayerNorm([channels, 11, 11]),
            nn.Sigmoid()  # 新增激活层
        )
        self.cat_conv = nn.Conv2d(2*channels, channels, 3, padding=1)
        self.cat_conv1 = nn.Conv2d(2*channels, channels, 3, padding=1)
        self.spatial_attention = MultiScaleSpatialAttention(channels)
    def forward(self, x, c, t):
        """
        输入:
        x: 扩散特征 [B,1152,11,11] 
        c: 条件特征 [B,1152,11,11]
        t: 时间步嵌入 [B,t_dim]
        """
        ####################################
        # 第一步：交叉显著性增强
        ####################################
        # 特征相乘突出共同显著性
        multiply = x * c  # [B,1152,11,11]
        
        # 归一化处理
        spatialmap = self.spatial_attention(multiply)
        
        # 残差增强
        x_enhanced = x + x * spatialmap  # [B,1152,11,11]
        c_enhanced = c + c * spatialmap  # [B,1152,11,11]

        ####################################
        # 第二步：协同通道注意力
        ####################################
        # 拼接增强特征
        combined = torch.cat([x_enhanced, c_enhanced], dim=1)  # [B,2304,11,11]
        combined = self.cat_conv(combined)
        # 生成通道注意力
        ch_attn = self.channel_attn(combined)  # [B,1152,1,1]
        

        # 通道注意力调制
        x_modulated = x_enhanced * ch_attn
        c_modulated = c_enhanced * ch_attn

        ####################################
        # 第三步：时间门控融合
        ####################################
        # 时间嵌入投影
        t_gate = self.t_proj(t)  # [B,1,11,11]
        cat_fusion = self.cat_conv1(torch.cat([x_modulated, c_modulated], dim=1))
        # 门控融合
        fused = cat_fusion + t_gate*x_modulated
        
        # 最终融合
        return fused  # [B,1152,11,11]

#rgbd融合
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LightCrossAttention(nn.Module):
    def __init__(self, in_dim_rgb, in_dim_depth, embed_dim=256, reduction_ratio=4):
        super().__init__()
        # Q/K投影层 (RGB和深度共享结构)
        self.proj_rgb = nn.Linear(in_dim_rgb, embed_dim * 2)  # 输出Q和K
        self.proj_depth = nn.Linear(in_dim_depth, embed_dim * 2)  # 输出Q和K
        
        # V生成：分别卷积后拼接
        self.v_conv_rgb = nn.Conv2d(in_dim_rgb, embed_dim, kernel_size=1)
        self.v_conv_depth = nn.Conv2d(in_dim_depth, embed_dim, kernel_size=1)
        self.cat_conv = nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1)
        # 差异注意力变换
        self.diff_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.norm = LayerNorm(embed_dim, LayerNorm_type='WithBias')
        # 融合层
        self.fusion_conv = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 2, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, rgb_feat, depth_feat):
        # 输入形状: RGB (B, C1, H, W), Depth (B, C2, H, W)
        B, _, H, W = rgb_feat.shape
        
        # 生成Q/K (B, H*W, embed_dim)
        rgb_qk = self.proj_rgb(rgb_feat.flatten(2).transpose(1, 2))  # B, H*W, 2*embed_dim
        rgb_q, rgb_k = torch.chunk(rgb_qk, 2, dim=-1)  # 各为B, H*W, embed_dim
        
        depth_qk = self.proj_depth(depth_feat.flatten(2).transpose(1, 2))
        depth_q, depth_k = torch.chunk(depth_qk, 2, dim=-1)
        
        # 生成V (B, embed_dim, H, W)
        v_rgb = self.v_conv_rgb(rgb_feat)  # B, embed_dim, H, W
        v_depth = self.v_conv_depth(depth_feat)  # B, embed_dim, H, W
        v = torch.cat([v_rgb, v_depth], dim=1)  # B, embed_dim*2, H, W
        v_fused = self.cat_conv(v)
        v1, v2 = torch.chunk(v_fused, 2, dim=1)  # 各为(B, embed_dim, H, W)
        # 路径1: RGB-Q 与 Depth-K 的相似性注意力
        rgb_q_fft = torch.fft.rfft2(rgb_q.float())
        depth_k_fft = torch.fft.rfft2(depth_k.float())
        attn_sim = rgb_q_fft*depth_k_fft
        attn_sim = torch.fft.irfft2(attn_sim, s=(H, W))
        attn_sim = attn_sim.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        attn_sim = self.norm(attn_sim)
        modulated_v1 = attn_sim * v1  # B, embed_dim, H, W
        
        # 路径2: Depth-Q 与 RGB-K 的差异注意力
        diff = torch.abs(depth_q - rgb_k)  # B, H*W, embed_dim
        diff = diff.transpose(1, 2).view(B, -1, H, W)  # B, embed_dim, H, W
        attn_diff = self.diff_conv(diff)  # B, embed_dim, H, W
        modulated_v2 = attn_diff * v2  # Hadamard乘积调制
        
        # 双路融合
        gate = self.fusion_gate(torch.cat([modulated_v1, modulated_v2], dim=1))
        output = gate[:, 0:1] * modulated_v1 + gate[:, 1:2] * modulated_v2
        return output

#fusion融合
class ProgressiveFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stage1: 144x88x88 → 288x44x44 (与特征2融合)
        self.stage1 = FusionBlock(in_ch_low=144, in_ch_high=288, out_ch=288)
        
        # Stage2: 288x44x44 → 576x22x22 (与特征3融合)
        self.stage2 = FusionBlock(in_ch_low=288, in_ch_high=576, out_ch=576)
        
        # Stage3: 576x22x22 → 1152x11x11 (与特征4融合)
        self.stage3 = FusionBlock(in_ch_low=576, in_ch_high=1152, out_ch=1152)
        
        self.convstage1 = ConvFusionBlock(in_ch_low=288, in_ch_high=576, out_ch=576)

        self.convstage2 = ConvFusionBlock(in_ch_low=576, in_ch_high=1152, out_ch=1152)
    def forward(self, f1, f2, f3, f4):
        
        # Stage1: 融合f1和f2 → 288x44x44
        fused_12 = self.stage1(f1, f2)
        
        # Stage2: 融合结果与f3 → 576x22x22
        fused_23 = self.stage2(f2, f3)
        
        # Stage3: 融合结果与f4 → 1152x11x11
        fused_34 = self.stage3(f3, f4)

        fused_123 = self.convstage1(fused_12, fused_23)
        output = self.convstage2(fused_123, fused_34)        
        return output


class FusionBlock(nn.Module):
    """ 相邻尺度融合单元（低分辨率特征 + 高分辨率特征） """
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        # 高分辨率特征处理：下采样 + 通道调整
        self.process_high = nn.Sequential(
            nn.Conv2d(in_ch_low, out_ch, 3, stride=2, padding=1),  # 空间下采样
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        # 低分辨率特征处理：通道调整
        self.process_low = nn.Sequential(
            nn.Conv2d(in_ch_high, out_ch, 1),  # 仅调整通道数
            nn.BatchNorm2d(out_ch)
        )
        
        # 自适应融合门控
        self.combinefusion = nn.Conv2d(2*out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, feat_high, feat_low):
        """
        输入:
            feat_low: 低分辨率特征 [B, in_ch_low, H, W]
            feat_high: 高分辨率特征 [B, in_ch_high, H, W]
        输出:
            融合后的特征 [B, out_ch, H/2, W/2]
        """
        # 处理低分辨率特征 (通道调整)
        low_proj = self.process_low(feat_low)  
        
        # 处理高分辨率特征 (下采样)
        high_proj = self.process_high(feat_high) 
        
        addfusion = low_proj+high_proj
        multipfusion = low_proj*high_proj
        
        # 生成融合权重
        combined = torch.cat([addfusion, multipfusion], dim=1)  # [B, 2*out_ch, H/2, W/2]
        
        # 加权融合
        fused = self.combinefusion(combined)
        return fused
    
class ConvFusionBlock(nn.Module):
    """ 相邻尺度融合单元（低分辨率特征 + 高分辨率特征） """
    def __init__(self, in_ch_low, in_ch_high, out_ch):
        super().__init__()
        # 高分辨率特征处理：下采样 + 通道调整
        self.process_high = nn.Sequential(
            nn.Conv2d(in_ch_low, out_ch, 3, stride=2, padding=1),  # 空间下采样
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        # 低分辨率特征处理：通道调整
        self.process_low = nn.Sequential(
            nn.Conv2d(in_ch_high, out_ch, 1),  # 仅调整通道数
            nn.BatchNorm2d(out_ch)
        )
        
        # 自适应融合门控
        self.combinefusion = nn.Conv2d(2*out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, feat_high, feat_low):
        """
        输入:
            feat_low: 低分辨率特征 [B, in_ch_low, H, W]
            feat_high: 高分辨率特征 [B, in_ch_high, H, W]
        输出:
            融合后的特征 [B, out_ch, H/2, W/2]
        """
        # 处理低分辨率特征 (通道调整)
        low_proj = self.process_low(feat_low)  
        
        # 处理高分辨率特征 (下采样)
        high_proj = self.process_high(feat_high) 
        
        
        # 生成融合权重
        combined = torch.cat([low_proj, high_proj], dim=1)  # [B, 2*out_ch, H/2, W/2]
        
        # 加权融合
        fused = self.combinefusion(combined)
        return fused

