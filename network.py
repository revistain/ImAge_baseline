import logging
import torch
import torch.nn.functional as F
from torch import nn
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
        
        self.backbone = get_backbone(args)
        self.learnable_aggregation_tokens = nn.Parameter(torch.zeros(1, args.num_learnable_aggregation_tokens, self.backbone.embed_dim)) if args.num_learnable_aggregation_tokens else None
        self.insertion_pos = 8 #args.freeze_te
        
        if args.freeze_te==12:
            self.learnable_aggregation_tokens.requires_grad = False
            print("aggregation_tokens are freeze")

        # feat_dim = self.backbone_rgb.embed_dim * args.num_learnable_aggregation_tokens  # 768*8=6144

    def _get_backbone_features(self, x: torch.Tensor, use_adapter: bool) -> torch.Tensor:
        backbone     = self.backbone
        agg_tokens   = self.learnable_aggregation_tokens
        film_adapter = getattr(backbone, 'film_adapter', None)

        x = backbone.prepare_tokens_with_masks(x)
        B = x.shape[0]
        for i in range(self.insertion_pos):
            x, _ = backbone.blocks[i](x, modality=use_adapter, film_adapter=film_adapter)
        x = torch.cat([agg_tokens.expand(B, -1, -1), x], dim=1)
        for i in range(self.insertion_pos, len(backbone.blocks)):
            x, _ = backbone.blocks[i](x, modality=use_adapter, film_adapter=film_adapter)
        x_norm = backbone.norm(x)
        return x_norm[:, :self.num_learnable_aggregation_tokens, :].flatten(1)

    def _forward_impl(self, x: torch.Tensor, use_adapter: bool) -> torch.Tensor:
        x_g  = self._get_backbone_features(x, use_adapter)
        return F.normalize(x_g, p=2, dim=-1)

    def compute_flow_loss(self, thermal_imgs: torch.Tensor, rgb_imgs: torch.Tensor) -> torch.Tensor:
        """Two-Phase OT-CFM flow matching loss (Phase1: CLS, Phase2: AGG)."""
        backbone     = self.backbone
        film_adapter = getattr(backbone, 'film_adapter', None)
        
        # 전체 레이어 수 (예: 12)와 AGG 토큰 삽입 위치 (예: 8)
        total_layers = len(backbone.blocks)
        insert_pos   = self.insertion_pos
        num_agg      = self.num_learnable_aggregation_tokens
        agg_tokens   = self.learnable_aggregation_tokens

        # ── Step 1: 두 modality를 adapter 없이 전체 레이어(total_layers) 통과 ──
        with torch.no_grad():
            h_thr = backbone.prepare_tokens_with_masks(self.norm_transform(thermal_imgs))
            h_rgb = backbone.prepare_tokens_with_masks(self.norm_transform(rgb_imgs))
            h_thr_list, h_rgb_list = [], []
            B = h_thr.shape[0]

            for i in range(total_layers):
                # 추론 코드(_get_backbone_features)와 동일하게 insertion_pos에서 AGG 토큰 삽입
                if i == insert_pos and agg_tokens is not None:
                    h_thr = torch.cat([agg_tokens.expand(B, -1, -1), h_thr], dim=1)
                    h_rgb = torch.cat([agg_tokens.expand(B, -1, -1), h_rgb], dim=1)

                h_thr, _ = backbone.blocks[i](h_thr, modality=False)
                h_rgb, _ = backbone.blocks[i](h_rgb, modality=False)
                h_thr_list.append(h_thr)   # [B, N, D]
                h_rgb_list.append(h_rgb)

        # ── Step 2: 각 레이어에서 Phase별 FM loss 계산 ─────────────────────
        flow_loss = thermal_imgs.new_zeros(())
        
        for i in range(total_layers):
            t       = i / max(total_layers - 1, 1)          # 0.0 → 1.0
            h_thr_l = h_thr_list[i]
            h_rgb_l = h_rgb_list[i]

            if i < insert_pos:
                # [Phase 1] 빌드업 구간 (Layer 0 ~ 7): CLS 토큰 매칭
                # 삽입 전이므로 CLS 토큰이 맨 앞(index 0)에 위치함
                target = h_rgb_l[:, 0] - h_thr_l[:, 0]
                x_t    = (1 - t) * h_thr_l[:, 0] + t * h_rgb_l[:, 0]
            else:
                # [Phase 2] 정밀 타격 구간 (Layer 8 ~ 11): AGG 토큰 매칭
                # torch.cat으로 AGG 토큰이 맨 앞에 삽입되었으므로 index 0 ~ num_agg-1 이 AGG 토큰임!
                # (CLS 토큰은 뒤로 밀려남)
                target = h_rgb_l[:, :num_agg] - h_thr_l[:, :num_agg]
                x_t    = (1 - t) * h_thr_l[:, :num_agg] + t * h_rgb_l[:, :num_agg]

            # adapter가 interpolated point를 입력받아 velocity 예측
            x_t_normed = backbone.blocks[i].norm2(x_t)
            
            if film_adapter is not None:
                pred = film_adapter(x_t_normed, i)
            else:
                pred = backbone.blocks[i].adapter(x_t_normed)

            # MSE 계산 (Phase2에서는 num_agg 개수의 토큰에 대해 평균 MSE가 적용됨)
            flow_loss = flow_loss + F.mse_loss(pred, target)

        return flow_loss / total_layers
    
    def forward(self, x: torch.Tensor, is_thermal=False) -> torch.Tensor:
        if isinstance(is_thermal, bool):
            is_thermal = torch.full((x.shape[0],), is_thermal, dtype=torch.bool, device=x.device)
        x = self.norm_transform(x)
        
        if is_thermal.all():
            return self._forward_impl(x, use_adapter=True)
        elif not is_thermal.any():
            return self._forward_impl(x, use_adapter=False)
        else:
            rgb_idx = (~is_thermal).nonzero(as_tuple=True)[0]
            thr_idx =   is_thermal .nonzero(as_tuple=True)[0]
            rgb_out = self._forward_impl(x[rgb_idx], use_adapter=False)
            thr_out = self._forward_impl(x[thr_idx], use_adapter=True)
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


def get_backbone(args):
    """GSV pretrained로 초기화, blocks >= freeze_te 만 trainable."""
    from backbone.vision_transformer import vit_small, vit_base, vit_large
    size = getattr(args, "backbone_size", "b")
    vit  = {"s": vit_small, "b": vit_base, "l": vit_large}[size]
    backbone = vit(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=4,
                   use_adapter=True, use_film_adapter=getattr(args, 'film_adapter', False))

    agg_tokens = None
    rgb_model_path = getattr(args, 'rgb_model_path', None)
    if not args.resume and rgb_model_path:
        ckpt = torch.load(rgb_model_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt)
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        backbone.load_state_dict({k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}, strict=False)
        logging.info(f"backbone initialized from: {rgb_model_path}")

    for p in backbone.parameters():
        p.requires_grad = False
        
    for name, child in backbone.blocks.named_children():
        if args.freeze_te and int(name) >= args.freeze_te:
            for p in child.parameters():
                p.requires_grad = True
                
    for p in backbone.norm.parameters():
        p.requires_grad = True
        
    for name, param in backbone.blocks.named_parameters():
        if 'adapter' in name.lower():
            param.requires_grad = True

    # FiLM 공유 adapter 파라미터도 학습 가능하게
    if backbone.film_adapter is not None:
        for param in backbone.film_adapter.parameters():
            param.requires_grad = True
            


    return backbone
