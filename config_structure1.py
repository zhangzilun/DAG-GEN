# config_structure1.py
from pathlib import Path

# ===== Bucket by number of layers L =====
# 4 buckets: (2-3), (4-5), (6-7), (8+)
BUCKETS = [
    ("B1_L_2_3", 2, 3),
    ("B2_L_4_5", 4, 5),
    ("B3_L_6_7", 6, 7),
    ("B4_L_8_plus", 8, 10**9),
]

# ===== Training artifacts =====
CHECKPOINT_DIR = Path("checkpoints_structure1")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PRINT_EVERY = 1
SAVE_EVERY = 0  # >0 to save periodic ckpts

# ===== Loss weights (structure-only) =====
W_BCE = 1.0


W_DEG_COV = 0.05
W_SRC_SINK_SOFT = 0.02

W_DAG = 0.0
W_LONGEST = 0.05

# soft source/sink params
SRC_SINK_TAU = 0.1
SRC_SINK_K = 10.0

