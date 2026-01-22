# config.py
from pathlib import Path
import torch

# === Paths ===
# 这里改成你自己的 gpickle 数据集路径
DATA_DIR = Path(r"E:/DAG-GNN/data/gpickle2")

CHECKPOINT_DIR = Path("checkpoints5")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# === Training params ===
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
PRINT_EVERY = 1          # 每多少个 epoch 打印一次
SAVE_EVERY = 0           # 大于 0 时每 SAVE_EVERY 个 epoch 存一次模型

# === Graph normalization (用来构造 s_vec 的归一化常数) ===
# 大概范围：你可以之后再根据自己的数据调整
NORM_N = 100.0   # 最大节点数
NORM_E = 400.0   # 最大边数
NORM_L = 10.0    # 最大层数
NORM_W = 10.0    # 最大层宽
NORM_T = 300.0   # 最大总执行时间 / 最长路径时间

# === 结构 & 时间相关常数 ===
EDGE_TIME_MIN = 1.0               # 预测时间的最小值，防止为 0
MAX_LONGEST_PATH_TIME = 300.0     # 用于 longest path penalty

# === Loss 权重（可以之后慢慢调） ===
W_BCE = 1.0            # 边 BCE
W_TIME = 0.2           # 边时间 L1
W_TOTALT = 0.05        # 总执行时间
W_LONGEST = 0.05       # 最长路径时间
W_DAG = 0.1            # DAG 正则（h(A)）
W_DEG_COV = 0.2        # 度覆盖 + 层级度分布 + 孤立点
W_SRC_SINK_SOFT = 0.05 # 源/汇软约束
W_TIME_NODE = 0.5      # 节点级平均时间对齐（风格）
W_NODE_TIME_UNI = 0.2  # 同一节点出边时间均匀性

# === 源 / 汇 soft 约束参数 ===
SRC_SINK_TAU = 0.1   # “接近 0 入度/出度”的阈值
SRC_SINK_K = 10.0    # Sigmoid 的斜率（越大越接近硬阈值）

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
