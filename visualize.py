#!/usr/bin/env python3
"""
Visualization script: creates a video showing thermal query (left, large) and
top-5 RGB database retrievals (right, stacked vertically).
  - Green border : correct match (within positive threshold)
  - Red border   : incorrect match

Usage example:
    python visualize.py \
        --resume <ckpt.pth> \
        --datasets_folder <path> \
        --dataset_name <name> \
        --test_seq Campus \
        --img_time allday \
        --output_video vis.mp4 \
        --fps 3 \
        --max_queries 200
"""

import os
import sys
import cv2
import torch
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import parser as arg_parser
import util
import commons
import datasets_ws
import network

import warnings
warnings.filterwarnings("ignore")

# ── Layout constants ──────────────────────────────────────────────────────────
# 원본 비율:  Thermal 640×256 (2.5:1),  RGB 1224×384 (3.19:1)
# QUERY_H = 400 → QUERY_W = 400 × 2.5 = 1000
# RESULT_H = 400 / 5 = 80 → RESULT_W = 80 × 3.19 ≈ 256
# Frame: (8+1000+8+256+8) × (8+400+8) = 1280 × 416
TOP_K     = 5
PAD       = 8
BORDER    = 5

QUERY_H   = 400
QUERY_W   = 1000   # 2.5 : 1  (Thermal 원본 비율)

RESULT_H  = QUERY_H // TOP_K   # 80px
RESULT_W  = 256                 # ≈ 80 × 3.19  (RGB 원본 비율)

COLOR_OK  = (80, 200,  80)   # BGR green
COLOR_NG  = (60,  60, 220)   # BGR red
COLOR_BG  = (30,  30,  30)

FONT      = cv2.FONT_HERSHEY_SIMPLEX

FRAME_W   = PAD + QUERY_W + PAD + RESULT_W + PAD   # 1280
FRAME_H   = PAD + QUERY_H + PAD                     # 416


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_display_img(path, dataset, is_thermal: bool, w: int, h: int) -> np.ndarray:
    """Load image for display (no model normalization) → BGR (h x w)."""
    if is_thermal:
        pil = dataset.get_thermal_img(path)
    else:
        pil = dataset.get_rgb_img(path)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (w, h))


def overlay_text(img, text, ok: bool):
    """텍스트를 이미지 좌하단에 오버레이."""
    h, w = img.shape[:2]
    color = COLOR_OK if ok else COLOR_NG
    # 반투명 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 22), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (5, h - 6), FONT, 0.48, color, 1, cv2.LINE_AA)


def make_frame(query_img, result_imgs, is_correct, q_idx: int) -> np.ndarray:
    """Compose one video frame: large query on left, top-K stacked on right."""
    frame = np.full((FRAME_H, FRAME_W, 3), COLOR_BG, dtype=np.uint8)

    # ── Query (left, large) ────────────────────────────────────────────────
    qx, qy = PAD, PAD
    frame[qy:qy + QUERY_H, qx:qx + QUERY_W] = query_img
    # 쿼리 라벨 (하단 오버레이)
    q_roi = frame[qy:qy + QUERY_H, qx:qx + QUERY_W]
    overlay = q_roi.copy()
    cv2.rectangle(overlay, (0, QUERY_H - 28), (QUERY_W, QUERY_H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, q_roi, 0.45, 0, q_roi)
    cv2.putText(q_roi, f"Query #{q_idx}  [Thermal]",
                (8, QUERY_H - 9), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Top-K results (right, vertical) ───────────────────────────────────
    rx = PAD + QUERY_W + PAD
    for k, (rimg, ok) in enumerate(zip(result_imgs, is_correct)):
        ry = PAD + k * RESULT_H

        cell = rimg.copy()
        border_color = COLOR_OK if ok else COLOR_NG
        cv2.rectangle(cell, (0, 0), (RESULT_W - 1, RESULT_H - 1), border_color, BORDER)

        # 텍스트 오버레이
        label = f"Top-{k + 1}  {'[O]' if ok else '[X]'}"
        overlay_text(cell, label, ok)

        frame[ry:ry + RESULT_H, rx:rx + RESULT_W] = cell

    return frame


# ── Frame data loader (멀티스레드용) ──────────────────────────────────────────

def load_frame_data(q_idx, eval_ds, predictions, positives_per_query):
    """한 프레임에 필요한 이미지들을 모두 로드해서 반환."""
    q_path = eval_ds.queries_paths[q_idx]
    q_img  = load_display_img(q_path, eval_ds, is_thermal=True,  w=QUERY_W, h=QUERY_H)

    positives   = set(positives_per_query[q_idx].tolist())
    result_imgs = []
    is_correct  = []
    for db_idx in predictions[q_idx]:
        r_path = eval_ds.database_paths[db_idx]
        result_imgs.append(
            load_display_img(r_path, eval_ds, is_thermal=False, w=RESULT_W, h=RESULT_H)
        )
        is_correct.append(int(db_idx) in positives)

    return q_img, result_imgs, is_correct


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(args, eval_ds, model):
    """Return (db_feats, query_feats) as float32 numpy arrays."""
    model.eval()
    all_feats = np.empty((len(eval_ds), args.features_dim), dtype="float32")

    with torch.no_grad():
        eval_ds.test_method = "hard_resize"
        db_loader = DataLoader(
            Subset(eval_ds, list(range(eval_ds.database_num))),
            batch_size=args.infer_batch_size,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        for imgs, idxs in tqdm(db_loader, desc="DB features   ", ncols=100):
            feats = model(imgs.to(args.device))
            all_feats[idxs.numpy()] = feats.cpu().numpy()

        q_loader = DataLoader(
            Subset(eval_ds, list(range(eval_ds.database_num,
                                       eval_ds.database_num + eval_ds.queries_num))),
            batch_size=args.infer_batch_size,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        for imgs, idxs in tqdm(q_loader, desc="Query features", ncols=100):
            feats = model(imgs.to(args.device), is_thermal=True)
            all_feats[idxs.numpy()] = feats.cpu().numpy()

    return all_feats[:eval_ds.database_num], all_feats[eval_ds.database_num:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Visualization-specific args를 sys.argv에서 먼저 분리
    VIS_FLAGS = {"--output_video", "--fps", "--max_queries", "--frame_step"}
    vis_argv, clean_argv = [], [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in VIS_FLAGS:
            vis_argv += [sys.argv[i], sys.argv[i + 1]]
            i += 2
        else:
            clean_argv.append(sys.argv[i])
            i += 1
    sys.argv = clean_argv

    args = arg_parser.parse_arguments()

    vis_parser = argparse.ArgumentParser(add_help=False)
    vis_parser.add_argument("--output_video", type=str, default="visualization_Campus.mp4")
    vis_parser.add_argument("--fps",          type=int, default=3)
    vis_parser.add_argument("--max_queries",  type=int, default=None)
    vis_parser.add_argument("--frame_step",   type=int, default=1,
                            help="매 N번째 쿼리만 시각화에 포함 (기본값: 1 = 전체)")
    vargs = vis_parser.parse_args(vis_argv)

    # Setup
    args.save_dir = join("test", args.save_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")

    # Model
    model = network.VPRmodel(args)
    model = model.to(args.device)
    if args.resume:
        logging.info(f"Loading checkpoint: {args.resume}")
        model = util.resume_model(args, model)
    model = torch.nn.DataParallel(model)
    model.eval()

    if args.aggregator:
        pass
    else:
        args.features_dim = (args.num_learnable_aggregation_tokens
                             * model.module.backbone.embed_dim)
    logging.info(f"features_dim: {args.features_dim}")

    # Dataset
    args.sequences = args.test_seq
    eval_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset_name, "test"
    )
    logging.info(f"Dataset: {eval_ds}")

    # Features & FAISS
    db_feats, q_feats = extract_features(args, eval_ds, model)
    index = faiss.IndexFlatL2(args.features_dim)
    index.add(db_feats)
    _, predictions = index.search(q_feats, TOP_K)

    positives_per_query = eval_ds.get_positives()

    # Video writer
    out_path = vargs.output_video
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, vargs.fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    n_queries = eval_ds.queries_num
    if vargs.max_queries is not None:
        n_queries = min(n_queries, vargs.max_queries)

    query_indices = list(range(0, n_queries, vargs.frame_step))
    logging.info(
        f"Rendering {len(query_indices)} frames (step={vargs.frame_step}, "
        f"total queries={n_queries}) → {out_path}  (fps={vargs.fps})"
    )

    # 멀티스레드로 이미지 미리 로드
    num_threads = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            q_idx: executor.submit(
                load_frame_data, q_idx, eval_ds, predictions, positives_per_query
            )
            for q_idx in query_indices
        }

        for q_idx in tqdm(query_indices, desc="Rendering", ncols=100):
            q_img, result_imgs, is_correct = futures[q_idx].result()
            frame = make_frame(q_img, result_imgs, is_correct, q_idx)
            writer.write(frame)

    writer.release()
    logging.info(f"Saved: {out_path}")
    print(f"\nDone. Video saved to: {out_path}")


if __name__ == "__main__":
    main()
