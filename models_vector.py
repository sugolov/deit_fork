# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import register_model
from timm.layers import trunc_normal_


__all__ = [
    'deit_base_patch4_32'
]

@register_model
def gs_base_patch4_32(pretrained=False, **kwargs):
    # NOTE: idk where these are added to kwargs but delete em
    # NOTE: used to work in old timm, but our implementation uses updated packages
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=4, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
        # NOTE: removed kwargs above, check that this DOES NOT affect accuracy
    model.default_cfg = {
        'url': '',
        'num_classes': 100,
        'input_size': (3, 32, 32),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def gs_base_patch2_32(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=2, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
        # NOTE: removed kwargs above, check that this DOES NOT affect accuracy
    model.default_cfg = {
        'url': '',
        'num_classes': 100,
        'input_size': (3, 32, 32),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model




@register_model
def gs_base_patch16_224(pretrained=False, **kwargs):
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
