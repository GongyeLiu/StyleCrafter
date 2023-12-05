import torch
import torch.nn as nn
from collections import OrderedDict
from lvdm.basics import (
    zero_module,
    conv_nd,
    avg_pool_nd
)
from einops import rearrange
from lvdm.modules.attention import register_attn_processor, set_attn_processor, DualCrossAttnProcessor, get_attn_processor
from lvdm.modules.attention import DualCrossAttnProcessorAS
from utils.utils import instantiate_from_config

from lvdm.modules.encoders.arch_transformer import Transformer


class StyleTransformer(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, num_heads=8, num_tokens=4, n_layers=2):
        super().__init__()
        scale = in_dim ** -0.5
        self.num_tokens = num_tokens
        self.style_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=in_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        self.proj = nn.Parameter(torch.randn(in_dim, out_dim) * scale)
    
    def forward(self, x):
        style_emb = self.style_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([style_emb, x], dim=1)
        # x = torch.cat([x, style_emb], dim=1)
        x = self.ln1(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_tokens, :])
        x = x @ self.proj
        return x


class ScaleEncoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1, num_heads=8, num_tokens=16, n_layers=2):
        super().__init__()
        scale = in_dim ** -0.5
        self.num_tokens = num_tokens
        self.scale_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=in_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)

        self.out = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
            nn.Tanh(),
        )
        
    def forward(self, x):
        scale_emb = self.scale_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([scale_emb, x], dim=1)
        x = self.ln1(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_tokens, :])
        x = self.out(x)
        return x


class DropPath(nn.Module):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep.
    """
    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p
    
    def forward(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args
        
        # params
        x = args[0]
        b = x.size(0)
        n = (torch.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=torch.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False
        
        # drop-path index
        index = torch.where(mask)[0]
        index = index[torch.randperm(len(index))[:n]]
        if zero is not None:
            index = torch.cat([index, torch.where(zero)[0]], dim=0)
        
        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output
    
    def broadcast(self, src, dst):
        assert src.size(0) == dst.size(0)
        shape = (dst.size(0), ) + (1, ) * (dst.ndim - 1)
        return src.view(shape)
    

class ImageContext(nn.Module):
    def __init__(self, width=1024, context_dim=768, token_num=1):
        super().__init__()
        self.width = width
        self.token_num = token_num
        self.context_dim = context_dim

        self.fc = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.SiLU(),
            nn.Linear(width, token_num * context_dim),
        )
        self.drop_path = DropPath(0.5)

    def forward(self, x):
        # x shape [B, C]
        out = self.drop_path(self.fc(x))
        out = rearrange(out, 'b (n c) -> b n c', n=self.token_num)
        return out


class StyleAdapterDualAttnAS(nn.Module):
    def __init__(self, image_context_config, scale_predictor_config, scale=1.0, use_norm=False, time_embed_dim=1024, mid_dim=32):
        super().__init__()
        self.image_context_model = instantiate_from_config(image_context_config)
        self.scale_predictor = instantiate_from_config(scale_predictor_config)
        self.scale = scale
        self.use_norm = use_norm
        self.time_embed_dim = time_embed_dim
        self.mid_dim = mid_dim
        
    def create_cross_attention_adapter(self, unet):
        ori_processor = register_attn_processor(unet)
        dual_attn_processor = {}
        for idx, key in enumerate(ori_processor.keys()):
            kv_state_dicts = {
                'k': {'weight': unet.state_dict()[key[:-10] + '.to_k.weight']},
                'v': {'weight': unet.state_dict()[key[:-10] + '.to_v.weight']},
            }
            context_dim = kv_state_dicts['k']['weight'].shape[1]
            inner_dim = kv_state_dicts['k']['weight'].shape[0]
            print(key, context_dim, inner_dim)
            
            dual_attn_processor[key] = DualCrossAttnProcessorAS(
                context_dim=context_dim,
                inner_dim=inner_dim,
                state_dict=kv_state_dicts,
                scale=self.scale,
                use_norm=self.use_norm,
                layer_idx=idx,
            )
    
        set_attn_processor(unet, dual_attn_processor)

        dual_attn_processor = {key.replace('.', '_'): value for key, value in dual_attn_processor.items()}
        self.add_module('kv_attn_layers', nn.ModuleDict(dual_attn_processor))
            
    def set_cross_attention_adapter(self, unet):
        dual_attn_processor = get_attn_processor(unet)
        for key in dual_attn_processor.keys():
            module_key = key.replace('.', '_')
            dual_attn_processor[key] = self.kv_attn_layers[module_key]
            print('set', key, module_key)
        set_attn_processor(unet, dual_attn_processor)

    def forward(self, x):
        # x shape [B, C]
        return self.image_context_model(x)
