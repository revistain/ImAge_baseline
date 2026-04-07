
import os
import torch
import parser
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from os.path import join
from datetime import datetime
from sklearn.manifold import TSNE

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
######################################### MODEL #########################################

model = network.VPRmodel(args)
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

model = torch.nn.DataParallel(model)

if args.aggregator:
    pass  # features_dim already set inside get_aggregator()
else:
    args.features_dim = args.num_learnable_aggregation_tokens * model.module.backbone.embed_dim
logging.info(f"features_dim: {args.features_dim}")

######################################### FEATURE EXTRACTION #########################################
all_features  = []
all_modalities = []
all_seq_labels = []
all_conditions = []

for seq in args.test_seq:
    args.sequences = [seq]
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Dataset: {test_ds}")

    recalls, recalls_str, db_feats, q_feats = test.test(
        args, test_ds, model, args.test_method, return_features=True
    )
    logging.info(f"Recalls on {seq}: {recalls_str}")

    # 메타데이터: [db_n + q_n] 순서로 정렬됨
    modalities, seq_labels, conditions = test_ds.get_metadata()

    # feature와 메타데이터 순서 맞추기: db 먼저, 그 다음 queries
    feats = np.concatenate([db_feats, q_feats], axis=0)
    all_features.append(feats)
    all_modalities.append(modalities)
    all_seq_labels.append(seq_labels)
    all_conditions.append(conditions)

all_features   = np.concatenate(all_features,   axis=0)
all_modalities = np.concatenate(all_modalities, axis=0)
all_seq_labels = np.concatenate(all_seq_labels, axis=0)
all_conditions = np.concatenate(all_conditions, axis=0)

logging.info(f"Total features: {all_features.shape}  "
             f"(RGB: {(all_modalities=='RGB').sum()}, Thermal: {(all_modalities=='Thermal').sum()})")

######################################### SUBSAMPLING #########################################
MAX_SAMPLES = 6000
if len(all_features) > MAX_SAMPLES:
    logging.info(f"Subsampling {len(all_features)} → {MAX_SAMPLES} points for t-SNE")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_features), MAX_SAMPLES, replace=False)
    all_features   = all_features[idx]
    all_modalities = all_modalities[idx]
    all_seq_labels = all_seq_labels[idx]
    all_conditions = all_conditions[idx]

######################################### t-SNE #########################################
logging.info("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
embeddings = tsne.fit_transform(all_features)
logging.info("t-SNE done.")

######################################### PLOT #########################################
MODALITY_COLORS  = {'RGB': '#2196F3', 'Thermal': '#FF5722'}
MODALITY_MARKERS = {'RGB': 'o',       'Thermal': '^'}

SEQ_COLORS = {
    'Campus':      '#4CAF50',
    'Residential': '#9C27B0',
    'Urban':       '#FF9800',
    'KAIST':       '#4CAF50',
    'SNU':         '#9C27B0',
    'Valley':      '#FF9800',
}

COND_COLORS = {
    'database':  '#607D8B',
    'morning':   '#FFC107',
    'clearsky':  '#03A9F4',
    'rainy':     '#00BCD4',
    'nighttime': '#1A237E',
    'afternoon': '#FF9800',
    'evening':   '#7B1FA2',
}

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(
    "t-SNE of ImAge Features — RGB pretrained / RGB-T evaluation\n"
    f"(n={len(embeddings)}, feat_dim={args.features_dim}, seqs={args.test_seq}, img_time={args.img_time})",
    fontsize=12
)

# ── Subplot 1: Modality ──────────────────────────────────────────────────────
ax = axes[0]
for mod in ['RGB', 'Thermal']:
    mask = all_modalities == mod
    ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
               c=MODALITY_COLORS[mod], marker=MODALITY_MARKERS[mod],
               s=8, alpha=0.4, linewidths=0, label=mod)
ax.set_title("Colored by Modality", fontsize=11)
ax.legend(markerscale=3, fontsize=9)
ax.set_xticks([]); ax.set_yticks([])

# ── Subplot 2: Sequence (shape = modality) ───────────────────────────────────
ax = axes[1]
present_seqs = [s for s in SEQ_COLORS if s in np.unique(all_seq_labels)]
for seq in present_seqs:
    for mod, marker in MODALITY_MARKERS.items():
        mask = (all_seq_labels == seq) & (all_modalities == mod)
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=SEQ_COLORS[seq], marker=marker,
                   s=8, alpha=0.4, linewidths=0)

ax.set_title("Colored by Sequence  (○=RGB, △=Thermal)", fontsize=11)
seq_patches = [mpatches.Patch(color=SEQ_COLORS[s], label=s) for s in present_seqs]
mod_handles = [
    mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=6, label='RGB'),
    mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=6, label='Thermal'),
]
ax.legend(handles=seq_patches + mod_handles, fontsize=8, markerscale=1.5)
ax.set_xticks([]); ax.set_yticks([])

# ── Subplot 3: Time Condition (shape = modality) ─────────────────────────────
ax = axes[2]
present_conds = [c for c in COND_COLORS if c in np.unique(all_conditions)]
for cond in present_conds:
    mod = 'RGB' if cond == 'database' else 'Thermal'
    marker = MODALITY_MARKERS[mod]
    mask = all_conditions == cond
    ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
               c=COND_COLORS[cond], marker=marker,
               s=8, alpha=0.4, linewidths=0, label=cond)

ax.set_title("Colored by Time Condition  (○=RGB DB, △=Thermal Query)", fontsize=11)
ax.legend(markerscale=3, fontsize=8)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out_path = join(args.save_dir, "tsne.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
logging.info(f"Saved t-SNE plot → {out_path}")
plt.show()
