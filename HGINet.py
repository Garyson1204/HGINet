import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.utils import get_root_logger
from timm.models.layers import LayerNorm2d
from mmcv.runner import load_checkpoint
import math
from collections import OrderedDict
import numpy as np
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_

from bra_legacy import BiLevelRoutingAttention

from _common import Attention, AttentionLePE, DWConv
from Transformer import Transformer


class Conv_RTFA(nn.Module):
    def __init__(self, in_chs):
        super(Conv_RTFA, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_chs[0], in_chs[0]*2, 3, padding=1, stride=2, bias=False),
                                 nn.BatchNorm2d(in_chs[0]*2),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs[0]*2, in_chs[0]*4, 3, padding=1, stride=2, bias=False),
                                 nn.BatchNorm2d(in_chs[0]*4),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs[0]*4, 768, 1, bias=False),
                                 nn.BatchNorm2d(768),
                                 nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_chs[1], in_chs[1]*2, 3, padding=1, stride=2, bias=False),
                                 nn.BatchNorm2d(in_chs[1]*2),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs[1]*2, 768, 1, bias=False),
                                 nn.BatchNorm2d(768),
                                 nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_chs[2], 768, 1, bias=False),
                                 nn.BatchNorm2d(768),
                                 nn.ReLU(inplace=True))
    
    def forward(self, feats):
        B = feats[0].shape[0]
        new_feats = []
        new_feat1 = self.conv1(feats[0]) # [B, 768, 32, 32]
        new_feats.append(new_feat1)
        new_feat2 = self.conv2(feats[1])
        new_feats.append(new_feat2)
        new_feat3 = self.conv3(feats[2])
        new_feats.append(new_feat3)
        new_feat4 = F.interpolate(feats[3], (32,32), mode='bilinear', align_corners=True)
        new_feats.append(new_feat4)
        return new_feats
    
    
class Conv_HGIT(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv_HGIT, self).__init__()
        
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.up2x_3_1 = UpSampling2x(in_chs, in_chs)
        self.up2x_3_2 = UpSampling2x(in_chs, in_chs)

    
    def forward(self, feats):
        feats[0] = self.conv1_1(feats[0]) # [B, 768, 16, 16]
        feats[1] = self.conv1_2(feats[1]) # [B, 768, 16, 16]
        feats[4] = self.conv3_1(feats[4]) 
        feats[4] = self.up2x_3_1(feats[4])
        feats[5] = self.conv3_2(feats[5])
        feats[5] = self.up2x_3_2(feats[5])
        return feats
    
    
class Conv_Decoder(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv_Decoder, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(in_chs),
                                     nn.ReLU(inplace=True))
        self.up2x = UpSampling2x(in_chs, out_chs)
    
    def forward(self, feats):
        feats = self.conv1(feats)
        feats = self.up2x(feats)
        return feats


class UpSampling2x(nn.Module): 
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, features):
        return self.up_module(features)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, start=False):  # 768, 384
        super(GroupFusion, self).__init__()
        temp_chs = in_chs
        if start:
            in_chs = in_chs
        else:
            in_chs *= 2

        self.gf1 = nn.Sequential(nn.Conv2d(in_chs, temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))

        self.gf2 = nn.Sequential(nn.Conv2d((temp_chs + temp_chs), temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))
        self.up2x = UpSampling2x(temp_chs, out_chs)

    def forward(self, f_r, f_l):
        f_r = self.gf1(f_r)  # chs 768
        f12 = self.gf2(torch.cat((f_r, f_l), dim=1))  # chs 768
        return f12, self.up2x(f12)


class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                     )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x



class RTFA(nn.Module):
    def __init__(self, depth=[4, 4, 18, 4], in_chans=3, num_classes=2, embed_dim=[96, 192, 384, 768],
                 head_dim=32, qk_scale=None, representation_size=None,
                 drop_path_rate=0.3, drop_rate=0.,
                 use_checkpoint_stages=[],
                 ######## 
                 n_win=8,
                 kv_downsample_mode='identity',
                 kv_per_wins=[-1, -1, -1, -1],
                 topks=[1, 4, 16, -2],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[96, 192, 384, 768],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=True,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[3, 3, 3, 3],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 pretrained=None, **kwargs
                 ):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            if (pe is not None) and i+1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i+1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j], 
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        # self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        # self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        
    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')   

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)

    def forward(self, x:torch.Tensor):
        return self.forward_features(x)
    

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1./ math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Projection(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(Projection, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input
        self.phi = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_phi = nn.BatchNorm2d(dim)
        self.theta = nn.Conv2d(dim, node_num, kernel_size=1, bias=False)
        self.bn_theta = nn.BatchNorm2d(node_num)


    def forward(self, x):
        B, C, H, W = x.size()
        phi = self.phi(x) # B,C,H,W
        phi = self.bn_phi(phi)
        phi = phi.view(B,C,-1).contiguous() # B,C,(HW)
        
        theta = self.theta(x) # B,N,H,W
        theta = self.bn_theta(theta)
        theta = theta.view(B,self.node_num,-1).contiguous() # B,N,(HW)
        nodes = torch.matmul(phi, theta.permute(0, 2, 1).contiguous())
        return nodes, theta


class TransformerLayer(nn.Module):
    def __init__(self, dim, loop):
        super(TransformerLayer, self).__init__()
        self.gt1 = MultiHeadAttentionLayer(hid_dim=dim, n_heads=8, dropout=0.1)
        self.gt2 = MultiHeadAttentionLayer(hid_dim=dim, n_heads=8, dropout=0.1)
        self.gts = [self.gt1, self.gt2]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gts = self.gts[0:loop]

    def forward(self, x):
        for gt in self.gts:
            x = gt(x) # b x c x k
        return x


class GraphTransformer(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.gt = TransformerLayer(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    #graph0: edge, graph1/2: region, assign:edge
    def forward(self, query, key, value, assign):
        # Hierarchical Graph Interaction
        m = self.corr_matrix(query, key, value)
        interacted_graph = query + m
        
        # Transformer Learning 
        enhanced_graph = self.gt(interacted_graph)
        enhanced_feat = enhanced_graph.bmm(assign) # reprojection
        enhanced_feat = self.conv(enhanced_feat.unsqueeze(3)).squeeze(3)
        return enhanced_feat

    def corr_matrix(self, query, key, value):
        assign = query.permute(0, 2, 1).contiguous().bmm(key)
        assign = F.softmax(assign, dim=-1) #normalize region-node
        m = assign.bmm(value.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m


class HGIT(nn.Module): 
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=768, num_clusters=1, dropout=0.1):
        super(HGIT, self).__init__()

        self.dim = dim

        self.proj_1 = Projection(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.proj_2 = Projection(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.conv1_1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1_2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        self.conv2_1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv2_2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        
        self.conv3 = nn.Sequential(BasicConv2d(self.dim*2, self.dim, BatchNorm, kernel_size=1, padding=0))
        
        self.gt1 = GraphTransformer(self.dim, BatchNorm, dropout) 
        self.gt2 = GraphTransformer(self.dim, BatchNorm, dropout) 

    def forward(self, feat1, feat2):
        # project features to graphs
        graph_2, assign_2 = self.proj_2(feat2)
        graph_1, assign_1 = self.proj_1(feat1)

        q_graph1 = self.conv1_1(graph_1.unsqueeze(3)).squeeze(3) # Q
        k_graph1 = self.conv1_2(graph_1.unsqueeze(3)).squeeze(3) # K

        # Hierarchical Graph Interaction && Transformer Learning and Reprojection
        q_graph2 = self.conv2_1(graph_2.unsqueeze(3)).squeeze(3) # Q
        k_graph2 = self.conv2_2(graph_2.unsqueeze(3)).squeeze(3) # K
        
        mutual_V = torch.cat((graph_2, graph_1), dim=1)
        mutual_V_new = self.conv3(mutual_V.unsqueeze(3)).squeeze(3) # V

        enhanced_feat1 = self.gt1(q_graph1, k_graph2, mutual_V_new, assign_1)
        feat1 = feat1 + enhanced_feat1.view(feat1.size()).contiguous()
        
        enhanced_feat2 = self.gt2(q_graph2, k_graph1, mutual_V_new, assign_2)
        feat2 = feat2 + enhanced_feat2.view(feat2.size()).contiguous()

        return feat1, feat2


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, BatchNorm=nn.BatchNorm2d):
        super(MultiHeadAttentionLayer, self).__init__()
        self.nodes = 8
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.node_conv = nn.Conv1d(hid_dim, hid_dim, 1, bias=True)
        self.eigen_vec_conv = nn.Conv1d(self.nodes, hid_dim, 1, bias=True)

        self.fc_q = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        self.fc_k = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        self.fc_v = nn.Conv1d(hid_dim, hid_dim, 1, bias=False)
        
        self.fc_o = nn.Conv1d(self.nodes, self.nodes, 1, bias=False)
        
        self.w_1 = nn.Conv1d(hid_dim, hid_dim*4, 1) # position-wise
        self.w_2 = nn.Conv1d(hid_dim*4, hid_dim, 1) # position-wise
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(dropout)
        self.a4 = nn.ReLU(inplace=True)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def get_lap_vec(self, graph):
        device = graph.device
        batch = graph.shape[0]
        graph_t = graph.permute(0, 2, 1).contiguous() # transpose
        dense_adj = torch.matmul(graph_t, graph)
        dense_adj = self.a4(dense_adj)
        in_degree = dense_adj.sum(dim=1).view(batch, -1)
        dense_adj = dense_adj.detach().cpu().float().numpy()
        in_degree = in_degree.detach().cpu().float().numpy()
        number_of_nodes = self.nodes

        # Laplacian
        A = dense_adj
        N = []
        for i in range(0,batch):
            N1 = np.diag(in_degree[i].clip(1) ** -0.5)
            N.append(N1)
        N = np.array(N)
        L = np.eye(number_of_nodes) - N @ A @ N

        # (sorted) eigenvectors with numpy
        try:
            EigVal, EigVec = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            for i in L.shape[0]:
                filename = "laplacian_{}.txt".format(i)
                np.savetxt(filename, L[i])

        
        eigvec = torch.from_numpy(EigVec).float().to(device) 
        eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float().to(device) 
        return eigvec, eigval 
        
    
    def forward(self, qkv):
        
        batch_size = qkv.shape[0]
        device = qkv.device
        scale = self.scale.to(device)
        eigvec, eigval = self.get_lap_vec(qkv)
        qkv = self.node_conv(qkv)
        eigvec = self.eigen_vec_conv(eigvec)
        qkv = qkv + eigvec
        residual = qkv
        
        Q = self.fc_q(qkv)
        K = self.fc_k(qkv)
        V = self.fc_v(qkv)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        
        attention = self.dropout(torch.softmax(energy, dim = -1))
                
        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.dropout(self.fc_o(x))
        residual = residual.permute(0, 2, 1).contiguous()
        x = x + residual
        
        residual = x = self.layer_norm(x).permute(0, 2, 1).contiguous()
        #x = [batch size, query len, hid dim]

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class CAFF(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4):
        super(CAFF, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.tf = Transformer(depth=depth,
                              num_heads=1,
                              embed_dim=embed_dim,
                              mlp_ratio=3,
                              num_patches=num_patches)
        self.hw = hw
        self.inc = inc

    def forward(self, x, glbmap):
        B, C, _, _ = x.shape

        x = torch.cat([x, glbmap], dim=1)
        x = self.conv_fuse(x)
        # pred
        p1 = self.conv_p1(x)
        matt = self.sigmoid(p1)
        matt = matt * (1 - matt)
        matt = self.conv_matt(matt)
        fea = x * (1 + matt)

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
        fea = self.tf(fea, True)
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))

        return p2
    

class Fusion(nn.Module):

    def __init__(self, channels=64, r=4):
        super(Fusion, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class Model(nn.Module):
    def __init__(self, ckpt, img_size=512):  # change img size to 512
        super(Model, self).__init__()
        self.encoder = RTFA() # RTFA encoder
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location='cpu') # pretrain pth path
            msg = self.encoder.load_state_dict({k.replace('backbone.',''):v for k,v in ckpt['model'].items()}, strict=False)
            print(msg)

        self.img_size = img_size
        self.vit_chs = 768

        self.conv_rtfa = Conv_RTFA(in_chs=[96,192,384])
        self.conv_hgit = Conv_HGIT(in_chs=768, out_chs=384)
        self.conv_caff_fusion = Conv_Decoder(in_chs=384, out_chs=384)
        self.conv_fusion_1 = Conv_Decoder(in_chs=384, out_chs=384)
        self.conv_fusion_2_1 = Conv_Decoder(in_chs=384, out_chs=192)
        self.conv_fusion_2_2 = Conv_Decoder(in_chs=384, out_chs=192)
        self.hgit_0 = HGIT(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=8, dropout=0.1)
        self.hgit_1 = HGIT(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=8, dropout=0.1)
        self.hgit_2 = HGIT(nn.BatchNorm2d, dim=self.vit_chs, num_clusters=8, dropout=0.1)

        
        self.caff_coarse = CAFF(inc=768, outc=384, hw=16, embed_dim=768, num_patches=256)
        self.caff_mid = CAFF(inc=768, outc=384, hw=32, embed_dim=768, num_patches=1024)
        self.caff_grained = CAFF(inc=768, outc=384, hw=64, embed_dim=768, num_patches=4096)
        self.fusion_1_1 = Fusion(channels=384)
        self.fusion_1_2 = Fusion(channels=384)
        self.fusion_2 = Fusion(channels=192)
        
        self.out1 = OutPut(in_chs=384, scale=16)
        self.out2 = OutPut(in_chs=384, scale=8)
        self.out3 = OutPut(in_chs=192, scale=4)

    def pred_out(self, gpd_outs):
        return self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2])

    
    def forward(self, img):
        # B Seq
        B, C, H, W = img.size()
        # RTFA
        x = self.encoder(img) 
        # unify the feature size
        x = self.conv_rtfa(x)
        
        # HGIT
        feature = []
        out1, out2 = self.hgit_0(x[3], x[2])
        feature.append(out1)
        feature.append(out2)
        out1, out2 = self.hgit_1(x[2], x[1])
        feature.append(out1)
        feature.append(out2)
        out1, out2 = self.hgit_2(x[1], x[0])
        feature.append(out1)
        feature.append(out2) # [5,4,3,2,1,0]
        
        feature_new = self.conv_hgit(feature) # [B, 768, 16, 16] & [B, 768, 32, 32] & [B, 768, 64, 64]
        
        # decoder network with CAFF modules
        feat1 = self.caff_coarse(feature_new[0], feature_new[1]) # [384,32]
        feat2 = self.caff_mid(feature_new[2], feature_new[3]) # [384,64]
        feat3 = self.caff_grained(feature_new[4], feature_new[5]) # [384,128]
        gpd_outs = []
        feat1 = self.conv_caff_fusion(feat1) # [32->64]
        feat1_1 = self.fusion_1_1(feat1, feat2) # [384,32] 
        gpd_outs.append(feat1_1)
        feat1_1 = self.conv_fusion_1(feat1_1) # [384,32 -> 384,64]
        feat1_2 = self.fusion_1_2(feat1_1, feat3) # [384,64]
        gpd_outs.append(feat1_2)
        feat1_1 = self.conv_fusion_2_1(feat1_1) # [384,64 -> 192,128]
        feat1_2 = self.conv_fusion_2_2(feat1_2) # [384,64 -> 192,128]
        feat2_1 = self.fusion_2(feat1_1, feat1_2) # [192,128]
        gpd_outs.append(feat2_1)
        return self.pred_out(gpd_outs)
