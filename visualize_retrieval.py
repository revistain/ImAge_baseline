"""
visualize_retrieval.py

매 epoch 검증 후, 두 케이스를 시각화:
  1. top-100 안에도 정답 없는 query (hard failure) — 10개
  2. top-5 안에 정답 있는 query (easy success)   — 10개

레이아웃 (한 그림):
  각 row = [query(thermal)] + [top-5 retrieved(RGB)] + [GT positive(RGB)]
  - retrieved: 초록 테두리 = TP, 빨간 테두리 = FP
  - GT: 파란 테두리
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from util import Raw2Celsius


# ──────────────────────────────────────────────────────────────────────────────

def _load_rgb(path, size=(224, 224)):
    path = str(path).strip("[]'")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def _load_thermal(path, size=(224, 224), dataset_type='ms2', min_temp=-20, max_temp=60):
    path = str(path).strip("[]'")
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if dataset_type == 'ms2':
        img = Raw2Celsius(img)
        img = np.clip(img, min_temp, max_temp)
    elif dataset_type == 'nsavp':
        img = np.clip(img, 22500, 25000).astype(np.float32)
        img = (img - 22500) / 2500.0 * 255.0
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, size)
    return img


def _add_border(img_np, color, width=6):
    """img_np: [H, W, 3] uint8, color: (R,G,B)"""
    img = img_np.copy()
    img[:width, :] = color
    img[-width:, :] = color
    img[:, :width] = color
    img[:, -width:] = color
    return img


# ──────────────────────────────────────────────────────────────────────────────

def visualize_epoch(eval_ds, predictions, positives_per_query,
                    epoch_num, seq_name, save_dir, n_samples=10, seed=0):
    """
    Args:
        eval_ds          : BaseDataset (test split)
        predictions      : [Q, K] np.int64 — faiss search 결과 database indices
        positives_per_query : list of arrays, 각 query의 GT db indices
        epoch_num        : 현재 epoch (파일명 용)
        seq_name         : 시퀀스 이름 (파일명 용)
        save_dir         : 저장 경로
        n_samples        : 각 카테고리당 샘플 수
    """
    rng = random.Random(seed)
    Q = len(positives_per_query)

    # ── 케이스 분류 ──────────────────────────────────────────────────────────
    fail_idxs = []   # top-100 안에 정답 없음
    succ_idxs = []   # top-5 안에 정답 있음

    for q in range(Q):
        pos = positives_per_query[q]
        if len(pos) == 0:
            continue
        top100 = predictions[q, :100]
        top5   = predictions[q, :5]
        if not np.any(np.in1d(top100, pos)):
            fail_idxs.append(q)
        elif np.any(np.in1d(top5, pos)):
            succ_idxs.append(q)

    fail_sample = rng.sample(fail_idxs, min(n_samples, len(fail_idxs)))
    succ_sample = rng.sample(succ_idxs, min(n_samples, len(succ_idxs)))

    dataset_type = eval_ds.dataset_type
    min_temp = getattr(eval_ds, 'min_temp', -20)
    max_temp = getattr(eval_ds, 'max_temp',  60)

    IMG_SIZE = (160, 160)
    GREEN  = (0,   200,  80)
    RED    = (220,  30,  30)
    BLUE   = ( 30, 100, 220)
    GRAY   = (160, 160, 160)

    def make_grid(query_idxs, title):
        n = len(query_idxs)
        if n == 0:
            return None
        ncols = 7   # query + 5 retrieved + GT
        nrows = n
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 1.8, nrows * 1.8 + 0.4))
        fig.suptitle(title, fontsize=11, y=1.0)
        if nrows == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Query(Thr)", "Ret#1", "Ret#2", "Ret#3", "Ret#4", "Ret#5", "GT"]
        for c, ct in enumerate(col_titles):
            axes[0, c].set_title(ct, fontsize=7)

        for row, q in enumerate(query_idxs):
            pos = positives_per_query[q]

            # ── query (thermal) ──────────────────────────────────────────
            q_path = eval_ds.queries_paths[q]
            q_img  = _load_thermal(q_path, IMG_SIZE, dataset_type, min_temp, max_temp)
            q_img  = _add_border(q_img, GRAY)
            axes[row, 0].imshow(q_img)
            axes[row, 0].axis("off")

            # ── top-5 retrieved (RGB) ─────────────────────────────────────
            for col in range(1, 6):
                db_idx = predictions[q, col - 1]
                is_tp  = db_idx in pos
                border = GREEN if is_tp else RED
                db_path = eval_ds.database_paths[db_idx]
                db_img  = _load_rgb(db_path, IMG_SIZE)
                db_img  = _add_border(db_img, border)
                axes[row, col].imshow(db_img)
                axes[row, col].axis("off")
                rank_txt = f"#{col}"
                if is_tp:
                    rank_txt += " ✓"
                axes[row, col].set_xlabel(rank_txt, fontsize=6, labelpad=1)

            # ── GT (best positive: 가장 가까운 db image) ─────────────────
            # pos 중 predictions 전체에서 가장 높은 순위(낮은 index)
            pred_list = list(predictions[q])
            best_gt = None
            for db_idx in pred_list:
                if db_idx in pos:
                    best_gt = db_idx
                    break
            if best_gt is None:
                best_gt = pos[0]  # 그냥 첫 번째 GT
            gt_img = _load_rgb(eval_ds.database_paths[best_gt], IMG_SIZE)
            gt_img = _add_border(gt_img, BLUE)
            axes[row, 6].imshow(gt_img)
            axes[row, 6].axis("off")

        # 범례
        patches = [
            mpatches.Patch(color=np.array(GREEN)/255, label="TP"),
            mpatches.Patch(color=np.array(RED)/255,   label="FP"),
            mpatches.Patch(color=np.array(BLUE)/255,  label="GT"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3,
                   fontsize=8, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout()
        return fig

    os.makedirs(save_dir, exist_ok=True)

    # failure grid
    fig_fail = make_grid(
        fail_sample,
        f"[{seq_name}] Epoch {epoch_num} — TOP-100 FAILURE ({len(fail_idxs)}/{Q} queries)"
    )
    if fig_fail:
        path = os.path.join(save_dir, f"ep{epoch_num:03d}_{seq_name}_fail.png")
        fig_fail.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig_fail)

    # success grid
    fig_succ = make_grid(
        succ_sample,
        f"[{seq_name}] Epoch {epoch_num} — TOP-5 SUCCESS ({len(succ_idxs)}/{Q} queries)"
    )
    if fig_succ:
        path = os.path.join(save_dir, f"ep{epoch_num:03d}_{seq_name}_succ.png")
        fig_succ.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig_succ)

    print(f"  [vis] {seq_name} ep{epoch_num}: "
          f"fail={len(fail_idxs)}, succ={len(succ_idxs)}, "
          f"saved → {save_dir}")
