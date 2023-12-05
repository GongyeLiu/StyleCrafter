import os, random
from einops import rearrange, repeat

import torch
from utils.utils import instantiate_from_config
from lvdm.models.ddpm3d import LatentDiffusion
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.modules.attention import TemporalTransformer

class T2VAdapterDepth(LatentDiffusion):
    def __init__(self, depth_stage_config, adapter_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_stage = instantiate_from_config(depth_stage_config)
        self.adapter = instantiate_from_config(adapter_config)
        self.condtype = adapter_config.cond_name
        
        if 'pretrained' in adapter_config: 
            self.load_pretrained_adapter(adapter_config.pretrained)
        
        for param in self.depth_stage.parameters():
            param.requires_grad = False
    
    def prepare_midas_input(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        x_midas = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear')
        return x_midas

    @torch.no_grad()
    def get_batch_depth(self, x, target_size):
        # x: (b, c, t, h, w)
        # get depth image, reshape to target_size and normalize to [-1, 1]
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_midas = self.prepare_midas_input(x)
        cond_depth = self.depth_stage(x_midas)
        cond_depth = torch.nn.functional.interpolate(cond_depth, size=target_size, mode='bilinear')
        depth_min, depth_max = torch.amin(cond_depth, dim=[1, 2, 3], keepdim=True), torch.amax(cond_depth, dim=[1, 2, 3], keepdim=True)
        cond_depth = (cond_depth - depth_min) / (depth_max - depth_min + 1e-7)
        cond_depth = 2. * cond_depth - 1.
        cond_depth = rearrange(cond_depth, '(b t) c h w -> b c t h w', b=b, t=t)
        return cond_depth
    
    def load_pretrained_adapter(self, adapter_ckpt):
        # load pretrained adapter
        print(">>> Load pretrained adapter checkpoint.")
        try:
            state_dict = torch.load(adapter_ckpt, map_location="cpu")
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            self.adapter.load_state_dict(state_dict, strict=True)
        except:
            state_dict = torch.load(adapter_ckpt, map_location=f"cpu")
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model_state_dict = self.adapter.state_dict()
            n_unmatched = 0
            for n, p in model_state_dict.items():
                if p.shape != state_dict[n].shape:
                    state_dict.pop(n)
                    n_unmatched += 1
            model_state_dict.update(state_dict)
            self.adapter.load_state_dict(model_state_dict)
            print(f"Pretrained adapter IS NOT complete [{n_unmatched} units have unmatched shape].")


class T2IAdapterStyleAS(LatentDiffusion):
    def __init__(self, style_stage_config, adapter_config, *args, **kwargs):
        super(T2IAdapterStyleAS, self).__init__(*args, **kwargs)
        self.adapter = instantiate_from_config(adapter_config)
        self.condtype = adapter_config.cond_name
        ## adapter loading / saving paths
        self.style_stage_model = instantiate_from_config(style_stage_config)

        self.adapter.create_cross_attention_adapter(self.model.diffusion_model)
            
        if 'pretrained' in adapter_config:
            self.load_pretrained_adapter(adapter_config.pretrained)
        
        # freeze the style stage model  
        for param in self.style_stage_model.parameters():
            param.requires_grad = False
    
    def load_pretrained_adapter(self, pretrained):
        state_dict = torch.load(pretrained, map_location=f"cpu")
        
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
        self.adapter.load_state_dict(state_dict, strict=False)
        print('>>> adapter checkpoint loaded.')

    @torch.no_grad()
    def get_batch_style(self, batch_x):
        b, c, h, w = batch_x.shape
        cond_style = self.style_stage_model(batch_x)
        return cond_style
    
class T2VFintoneStyleAS(T2IAdapterStyleAS):
    def _get_temp_attn_parameters(self):
        temp_attn_params = []
        def register_recr(net_, name):
            if isinstance(net_, TemporalTransformer):
                temp_attn_params.extend(net_.parameters())
            else:
                for sub_name, net in net_.named_children():
                    register_recr(net, f"{name}.{sub_name}")
                
        for name, net in self.model.diffusion_model.named_children():
            register_recr(net, name)
        return temp_attn_params

    def _get_temp_attn_state_dict(self):
        temp_attn_state_dict = {}
        def register_recr(net_, name):
            if isinstance(net_, TemporalTransformer):
                temp_attn_state_dict[name] = net_.state_dict()
            else:
                for sub_name, net in net_.named_children():
                    register_recr(net, f"{name}.{sub_name}")
                
        for name, net in self.model.diffusion_model.named_children():
            register_recr(net, name)
        return temp_attn_state_dict

    def _load_temp_attn_state_dict(self, temp_attn_state_dict):
        def register_recr(net_, name):
            if isinstance(net_, TemporalTransformer):
                net_.load_state_dict(temp_attn_state_dict[name], strict=True)
            else:
                for sub_name, net in net_.named_children():
                    register_recr(net, f"{name}.{sub_name}")
                
        for name, net in self.model.diffusion_model.named_children():
            register_recr(net, name)

    def load_pretrained_temporal(self, pretrained):
        temp_attn_ckpt = torch.load(pretrained, map_location=f"cpu")
        if "state_dict" in list(temp_attn_ckpt.keys()):
            temp_attn_ckpt = temp_attn_ckpt["state_dict"]
        self._load_temp_attn_state_dict(temp_attn_ckpt)
        print('>>> Temporal Attention checkpoint loaded.')