import logging
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2

from aggregators.netvlad import NetVLAD
from aggregators.salad import SALAD
from aggregators.boq import BoQ


class VPRmodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch_name = args.backbone
        self.aggregator_name = args.aggregator
        self.aggregator = get_aggregator(args)
        self.num_learnable_aggregation_tokens = args.num_learnable_aggregation_tokens
        self.insertion_pos = args.insert_te
        self.norm_transform = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # RGB backbone: GSV pretrained, fully frozen
        self.backbone_rgb, agg_tokens_rgb = get_backbone_rgb(args)
        self.learnable_agg_tokens_rgb = nn.Parameter(agg_tokens_rgb.clone())
        self.learnable_agg_tokens_rgb.requires_grad = False

        # Thermal backbone: blocks 0~(freeze_te-1) frozen, blocks freeze_te~11 trainable
        self.backbone_thermal, agg_tokens_thermal = get_backbone_thermal(args)
        self.learnable_agg_tokens_thermal = nn.Parameter(agg_tokens_thermal.clone())
        self.learnable_agg_tokens_thermal.requires_grad = True

        feat_dim = self.backbone_rgb.embed_dim * args.num_learnable_aggregation_tokens  # 768*8=6144

    def _get_backbone_features(self, x: torch.Tensor, is_thermal: bool) -> torch.Tensor:
        backbone   = self.backbone_thermal   if is_thermal else self.backbone_rgb
        agg_tokens = self.learnable_agg_tokens_thermal if is_thermal else self.learnable_agg_tokens_rgb

        x = backbone.prepare_tokens_with_masks(x)
        B = x.shape[0]
        for i in range(self.insertion_pos):
            x = backbone.blocks[i](x)
        x = torch.cat([agg_tokens.expand(B, -1, -1), x], dim=1)
        for i in range(self.insertion_pos, len(backbone.blocks)):
            x = backbone.blocks[i](x)
        x_norm = backbone.norm(x)
        return x_norm[:, :self.num_learnable_aggregation_tokens, :].flatten(1)

    def _forward_impl(self, x: torch.Tensor, is_thermal: bool) -> torch.Tensor:
        x_g  = self._get_backbone_features(x, is_thermal)
        return F.normalize(x_g, p=2, dim=-1)

    def forward(self, x: torch.Tensor, is_thermal=False) -> torch.Tensor:
        if isinstance(is_thermal, bool):
            is_thermal = torch.full((x.shape[0],), is_thermal, dtype=torch.bool, device=x.device)
        x = self.norm_transform(x)
        
        if is_thermal.all():
            return self._forward_impl(x, is_thermal=True)
        elif not is_thermal.any():
            return self._forward_impl(x, is_thermal=False)
        else:
            rgb_idx = (~is_thermal).nonzero(as_tuple=True)[0]
            thr_idx =   is_thermal .nonzero(as_tuple=True)[0]
            rgb_out = self._forward_impl(x[rgb_idx], is_thermal=False)
            thr_out = self._forward_impl(x[thr_idx], is_thermal=True)
            feats   = torch.cat([rgb_out, thr_out])
            inv     = torch.empty(len(is_thermal), dtype=torch.long, device=x.device)
            inv[rgb_idx] = torch.arange(len(rgb_idx), device=x.device)
            inv[thr_idx] = len(rgb_idx) + torch.arange(len(thr_idx), device=x.device)
            return feats[inv]

# ──────────────────────────────────────────────────────────────────────────────

def get_aggregator(args):
    if not args.aggregator:
        return None
    elif args.aggregator == "netvlad":
        return NetVLAD(clusters_num=8, dim=768)
    elif args.aggregator == "salad":
        args.features_dim = 8448
        return SALAD(num_channels=768)
    elif args.aggregator == "boq":
        args.features_dim = 12288
        return BoQ(in_channels=768, proj_channels=384, num_layers=2, num_queries=64, row_dim=32)


def get_backbone_rgb(args):
    """GSV pretrained ImAge 로드, 완전 frozen. (adapter 없음)"""
    import backbone.dinov2.block as dinoblock
    dinoblock.adapter_dim = None

    from backbone.vision_transformer import vit_small, vit_base, vit_large
    size = getattr(args, "backbone_size", "b")
    vit  = {"s": vit_small, "b": vit_base, "l": vit_large}[size]
    rgb_backbone = vit(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=4, use_adapter=False)

    agg_tokens = None
    rgb_model_path = getattr(args, 'rgb_model_path', None)
    if not args.resume and rgb_model_path:
        ckpt = torch.load(rgb_model_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt)
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        rgb_backbone.load_state_dict(
            {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')},
            strict=True)
        if 'learnable_aggregation_tokens' in sd:
            agg_tokens = sd['learnable_aggregation_tokens']
        logging.info(f"RGB backbone loaded from: {rgb_model_path}")

    for p in rgb_backbone.parameters():
        p.requires_grad = False
    return rgb_backbone, agg_tokens

def get_backbone_thermal(args):
    """GSV pretrained로 초기화, blocks >= freeze_te 만 trainable."""
    import backbone.dinov2.block as dinoblock
    dinoblock.adapter_dim = None

    from backbone.vision_transformer import vit_small, vit_base, vit_large
    size = getattr(args, "backbone_size", "b")
    vit  = {"s": vit_small, "b": vit_base, "l": vit_large}[size]
    thr_backbone = vit(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=4, use_adapter=True)

    agg_tokens = None
    rgb_model_path = getattr(args, 'rgb_model_path', None)
    if not args.resume and rgb_model_path:
        ckpt = torch.load(rgb_model_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt)
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        thr_backbone.load_state_dict({k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}, strict=False)
        if 'learnable_aggregation_tokens' in sd:
            agg_tokens = sd['learnable_aggregation_tokens']
        logging.info(f"Thermal backbone initialized from: {rgb_model_path}")

    for p in thr_backbone.parameters():
        p.requires_grad = False
        
    for name, child in thr_backbone.blocks.named_children():
        if args.freeze_te and int(name) >= args.freeze_te:
            for p in child.parameters():
                p.requires_grad = True
    # for p in thr_backbone.norm.parameters():
    #     p.requires_grad = True
        
    for name, param in thr_backbone.blocks.named_parameters():
        if 'adapter' in name.lower():
            param.requires_grad = True
            


    return thr_backbone, agg_tokens
