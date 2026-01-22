# config.py
from pathlib import Path
import torch

# === Paths ===

DATA_DIR = Path(r"../src/gpickle2")

CHECKPOINT_DIR = Path("checkpoints5")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# === Training params ===
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
PRINT_EVERY = 1          
SAVE_EVERY = 0           

# === Graph normalization  ===
NORM_N = 100.0   
NORM_E = 400.0   
NORM_L = 10.0    
NORM_W = 10.0    
NORM_T = 300.0  


EDGE_TIME_MIN = 1.0              
MAX_LONGEST_PATH_TIME = 300.0     

# === Loss  ===
W_BCE = 1.0            
W_TIME = 0.2          
W_TOTALT = 0.05        
W_LONGEST = 0.05       
W_DAG = 0.1           
W_DEG_COV = 0.2        
W_SRC_SINK_SOFT = 0.05 
W_TIME_NODE = 0.5     
W_NODE_TIME_UNI = 0.2  


SRC_SINK_TAU = 0.1   
SRC_SINK_K = 10.0    

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


