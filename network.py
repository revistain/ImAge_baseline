import logging
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

from aggregators.netvlad import NetVLAD
from aggregators.salad import SALAD
from aggregators.boq import BoQ


class CFMTranslator(nn.Module):
    """OT-CFM velocity network: v_θ(x_t, t) ≈ x_1 - x_0.

    Training: MSE(v_θ((1-t)*x0 + t*x1, t), x1 - x0)  for t ~ U(0,1)
    Inference: Euler ODE solve  x1_hat = x0 + Σ dt * v_θ(x_t, t)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 64]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(self._sinusoidal_embed(t))  # [B, hidden]
        return self.net(torch.cat([x, t_emb], dim=-1))      # [B, feat_dim]

    def ode_solve(self, x0: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Euler integration: thermal feature → RGB-like feature."""
        x = x0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + dt * self.forward(x, t)
        return x


class VPRmodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch_name = args.backbone
        self.aggregator_name = args.aggregator
        self.aggregator = get_aggregator(args)
        self.num_learnable_aggregation_tokens = args.num_learnable_aggregation_tokens
        self.insertion_pos = 8
        self.norm_transform = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.cfm_steps = getattr(args, 'cfm_steps', 1)

        self.backbone = get_backbone(args)
        self.learnable_aggregation_tokens = nn.Parameter(
            torch.zeros(1, args.num_learnable_aggregation_tokens, self.backbone.embed_dim)
        ) if args.num_learnable_aggregation_tokens else None

        feat_dim = self.backbone.embed_dim * args.num_learnable_aggregation_tokens  # 768*8=6144
        self.cfm_translator = CFMTranslator(feat_dim, hidden_dim=2048)

    def _get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Both modalities use identical frozen backbone — no adapter."""
        backbone   = self.backbone
        agg_tokens = self.learnable_aggregation_tokens

        x = backbone.prepare_tokens_with_masks(x)
        B = x.shape[0]
        for i in range(self.insertion_pos):
            x, _ = backbone.blocks[i](x, modality=False)
        x = torch.cat([agg_tokens.expand(B, -1, -1), x], dim=1)
        for i in range(self.insertion_pos, len(backbone.blocks)):
            x, _ = backbone.blocks[i](x, modality=False)
        x_norm = backbone.norm(x)
        return x_norm[:, :self.num_learnable_aggregation_tokens, :].flatten(1)  # [B, feat_dim]

    def compute_fm_loss(self, thermal_imgs: torch.Tensor, rgb_imgs: torch.Tensor) -> torch.Tensor:
        """OT-CFM loss on descriptor space.

        x0 = thermal descriptor (frozen backbone, no grad)
        x1 = RGB descriptor    (frozen backbone, no grad)
        x_t = (1-t)*x0 + t*x1,  t ~ U(0,1)
        Loss = MSE(v_θ(x_t, t),  x1 - x0)
        """
        with torch.no_grad():
            x0 = self._get_backbone_features(self.norm_transform(thermal_imgs))
            x1 = self._get_backbone_features(self.norm_transform(rgb_imgs))

        B  = x0.shape[0]
        t  = torch.rand(B, device=x0.device)
        x_t    = (1 - t[:, None]) * x0 + t[:, None] * x1
        target = x1 - x0                               # constant velocity (OT-CFM)
        v_pred = self.cfm_translator(x_t, t)
        return F.mse_loss(v_pred, target)

    def forward(self, x: torch.Tensor, is_thermal=False) -> torch.Tensor:
        if isinstance(is_thermal, bool):
            is_thermal = torch.full((x.shape[0],), is_thermal, dtype=torch.bool, device=x.device)
        x = self.norm_transform(x)

        if is_thermal.all():
            raw = self._get_backbone_features(x)
            out = self.cfm_translator.ode_solve(raw, steps=self.cfm_steps)
            return F.normalize(out, p=2, dim=-1)
        elif not is_thermal.any():
            raw = self._get_backbone_features(x)
            return F.normalize(raw, p=2, dim=-1)
        else:
            rgb_idx = (~is_thermal).nonzero(as_tuple=True)[0]
            thr_idx =   is_thermal .nonzero(as_tuple=True)[0]
            rgb_raw = self._get_backbone_features(x[rgb_idx])
            rgb_out = F.normalize(rgb_raw, p=2, dim=-1)
            thr_raw = self._get_backbone_features(x[thr_idx])
            thr_out = F.normalize(self.cfm_translator.ode_solve(thr_raw, steps=self.cfm_steps), p=2, dim=-1)
            feats = torch.cat([rgb_out, thr_out])
            inv   = torch.empty(len(is_thermal), dtype=torch.long, device=x.device)
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
    """GSV pretrained backbone, fully frozen. No adapter — CFMTranslator handles modality gap."""
    from backbone.vision_transformer import vit_small, vit_base, vit_large
    size = getattr(args, "backbone_size", "b")
    vit  = {"s": vit_small, "b": vit_base, "l": vit_large}[size]
    backbone = vit(patch_size=14, img_size=518, init_values=1, block_chunks=0,
                   num_register_tokens=4, use_adapter=False, use_film_adapter=False)

    rgb_model_path = getattr(args, 'rgb_model_path', None)
    if not args.resume and rgb_model_path:
        ckpt = torch.load(rgb_model_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt)
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[7:]: v for k, v in sd.items()}
        backbone.load_state_dict(
            {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')},
            strict=False
        )
        logging.info(f"backbone initialized from: {rgb_model_path}")

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone
