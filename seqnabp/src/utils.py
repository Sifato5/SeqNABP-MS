import os, json, random, numpy as np, torch
LABELS={"nonNABP":0,"RBP":1,"DSB":2,"SSB":3}
NAMES={v:k for k,v in LABELS.items()}
def config(path):
    with open(path) as f: return json.load(f)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
