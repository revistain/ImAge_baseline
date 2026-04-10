"""
gap_aligner.py

Procrustes SVD top-k로 RGB-Thermal descriptor gap 정렬 축 관리.

분석 결과 요약:
  - gap은 ~50개 방향에 집중 (top-100에서 alignment peak, 이후 noise)
  - PCA(256) 전처리 후 [256x256] cross-covariance SVD → D=8448 직접 SVD 불필요
  - W_thr = P.T @ V_k  [D, k]: Thermal 투영 방향
  - W_rgb = P.T @ U_k  [D, k]: RGB     투영 방향
"""

import numpy as np
import torch
from sklearn.decomposition import PCA


class GapAligner:
    """
    매 epoch 끝에 Procrustes SVD top-k 방향을 계산하고,
    training_step에서 Thermal → RGB 방향 정렬 loss를 제공.

    Procrustes 분석 (Urban/Residential 실험 결과):
      - top-50 directions: alignment cosine 0.708 / 0.724  (~96% of full W* benefit)
      - top-100: peak (0.725 / 0.726)
      - top-256: 0.659 / 0.660  ← noise 포함시 오히려 하락
      → k=50 권장
    """

    def __init__(self, k: int = 50, pca_dim: int = 256):
        self.k       = k
        self.pca_dim = pca_dim

        # update() 이전에는 None → loss()가 0.0 반환
        self.W_thr:    torch.Tensor | None = None   # [D, k]
        self.W_rgb:    torch.Tensor | None = None   # [D, k]
        self.mean_thr: torch.Tensor | None = None   # [D]
        self.mean_rgb: torch.Tensor | None = None   # [D]
        self.sigma_k:  torch.Tensor | None = None   # [k]  (normalized weights)

    # ──────────────────────────────────────────────────────────────

    def update(self, f_rgb: np.ndarray, f_thr: np.ndarray) -> None:
        """
        Args:
            f_rgb : [N, D] L2-normalized RGB     descriptors (float32, numpy)
            f_thr : [N, D] L2-normalized Thermal descriptors (float32, numpy)
                    각 행은 동일 위치의 matched pair
        """
        N, D = f_rgb.shape
        pca_dim = min(self.pca_dim, N - 1, D)

        # 1. 각 modality의 mean 계산 후 각자 centering
        mean_rgb = f_rgb.mean(axis=0)   # [D]
        mean_thr = f_thr.mean(axis=0)   # [D]
        f_rgb_c  = f_rgb - mean_rgb     # [N, D]
        f_thr_c  = f_thr - mean_thr     # [N, D]

        # 2. PCA on combined centered data → 공분산 구조 파악 (mean 재적용 안 됨)
        pca = PCA(n_components=pca_dim, random_state=42)
        pca.fit(np.concatenate([f_rgb_c, f_thr_c], axis=0))   # [2N, D], already centered

        # PCA 방향으로만 투영 (mean은 이미 빠짐)
        P       = pca.components_          # [pca_dim, D]
        f_rgb_p = f_rgb_c @ P.T            # [N, pca_dim]
        f_thr_p = f_thr_c @ P.T            # [N, pca_dim]

        # 3. Cross-covariance SVD: M = F_rgb_p.T @ F_thr_p  [pca_dim, pca_dim]
        M = f_rgb_p.T @ f_thr_p
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # 4. Top-k axes, normalized weights
        k = min(self.k, len(S))
        w = S[:k] / S[:k].sum()

        # 5. Project directions back to original D-dim space
        #    pca.components_ = P : [pca_dim, D]
        #    W_thr = P.T @ V_k   : [D, k]  (Thermal alignment directions)
        #    W_rgb = P.T @ U_k   : [D, k]  (RGB    alignment directions)
        self.mean_thr = torch.tensor(mean_thr, dtype=torch.float32)  # [D]
        self.mean_rgb = torch.tensor(mean_rgb, dtype=torch.float32)  # [D]
        self.W_thr    = torch.tensor(P.T @ Vt[:k, :].T,          dtype=torch.float32)  # [D, k]
        self.W_rgb    = torch.tensor(P.T @ U[:, :k],              dtype=torch.float32)  # [D, k]
        self.sigma_k  = torch.tensor(w,                           dtype=torch.float32)  # [k]

        print(f"  [GapAligner] k={k}  N={N}  "
              f"expl_var={pca.explained_variance_ratio_.sum():.3f}  "
              f"top-5 S={np.round(S[:5], 1)}")

    # ──────────────────────────────────────────────────────────────

    def loss(self, f_thr: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        """
        Matched pair gap alignment loss.

        각 axis i에서 Thermal projection이 RGB projection과 같아지도록 당김.
        L_gap = mean_B [ sum_k sigma_k * (f_thr @ W_thr[:,k] - f_rgb @ W_rgb[:,k])^2 ]

        Args:
            f_thr : [B, D]  Thermal descriptors (gradient 흐름)
            f_rgb : [B, D]  RGB     descriptors (detached)

        Returns:
            scalar Tensor
        """
        if self.W_thr is None or self.mean_thr is None:
            return f_thr.new_tensor(0.0)

        dev      = f_thr.device
        mean_thr = self.mean_thr.to(dev)  # [D]
        mean_rgb = self.mean_rgb.to(dev)  # [D]
        W_t      = self.W_thr.to(dev)    # [D, k]
        W_r      = self.W_rgb.to(dev)    # [D, k]
        sigma    = self.sigma_k.to(dev)  # [k]

        # 각 modality의 중심으로 centering → project
        pt = (f_thr.float() - mean_thr) @ W_t   # [B, k]
        pr = (f_rgb.float() - mean_rgb) @ W_r   # [B, k]

        diff = pt - pr                       # [B, k]
        return (diff.pow(2) * sigma.unsqueeze(0)).sum(dim=1).mean()