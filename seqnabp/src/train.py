import os
import copy
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .model import SeqNABPMS
from .data_utils import EmbDataset, collate
from .evaluate import metrics
from .utils import device, seed_all


def compute_fused_probs(o):
    p_dir = o["logits"].softmax(-1)
    pn = o["nabp"].softmax(-1)
    pd_br = o["dbr"].softmax(-1)
    ps_sd = o["ssd"].softmax(-1)

    p_hier = torch.stack([
        pn[:, 0],
        pn[:, 1] * pd_br[:, 0],
        pn[:, 1] * pd_br[:, 1] * ps_sd[:, 0],
        pn[:, 1] * pd_br[:, 1] * ps_sd[:, 1]
    ], dim=-1)

    return 0.5 * p_dir + 0.5 * p_hier


def loss_fn(o, y):
    loss = nn.functional.cross_entropy(o["logits"], y)
    nb = (y != 0)
    db = (y == 2) | (y == 3)
    loss += 0.5 * nn.functional.cross_entropy(o["nabp"], nb.long())
    if nb.any():
        loss += 0.5 * nn.functional.cross_entropy(
            o["dbr"][nb], (y[nb] != 1).long()
        )
    if db.any():
        loss += 0.5 * nn.functional.cross_entropy(
            o["ssd"][db], (y[db] == 3).long()
        )
    return loss


def epoch(model, loader, opt, dev, train=True):
    model.train(train)
    total = 0.0
    ys = []
    ps = []

    for x, m, y, _ in loader:
        x, m, y = x.to(dev), m.to(dev), y.to(dev)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            o = model(x, m)
            loss = loss_fn(o, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        total += loss.item() * len(y)
        ys.extend(y.cpu().tolist())
        p_fused = compute_fused_probs(o)
        ps.extend(p_fused.detach().cpu().numpy())

    return total / len(loader.dataset), metrics(ys, ps)


def train_fold(safetensors_path, metadata, tr, va, fold, c, out,
               input_dim):
    seed_all(c["seed"] + fold)
    dev = device()

    tr_labels = np.asarray([metadata[pid]["label"] for pid in tr])
    tr_inner, stop_inner = train_test_split(
        np.asarray(tr),
        test_size=0.12,
        random_state=c["seed"] + fold,
        stratify=tr_labels
    )

    tl = DataLoader(
        EmbDataset(safetensors_path, metadata, tr_inner),
        batch_size=c["batch_size"],
        shuffle=True,
        collate_fn=collate
    )
    stop_loader = DataLoader(
        EmbDataset(safetensors_path, metadata, stop_inner),
        batch_size=c["batch_size"],
        shuffle=False,
        collate_fn=collate
    )
    test_loader = DataLoader(
        EmbDataset(safetensors_path, metadata, va),
        batch_size=c["batch_size"],
        shuffle=False,
        collate_fn=collate
    )

    model = SeqNABPMS(
        input_dim, c["hidden_dim"], c["num_heads"], c["dropout"]
    ).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=c["learning_rate"],
        weight_decay=c["weight_decay"]
    )

    best = -1.0
    state = None
    wait = 0
    hist = []

    for ep in range(c["max_epochs"]):
        trl, _ = epoch(model, tl, opt, dev, True)
        val, vm = epoch(model, stop_loader, opt, dev, False)
        hist.append({
            "epoch": ep + 1,
            "train_loss": trl,
            "val_loss": val,
            **{"val_" + k: v for k, v in vm.items()}
        })
        print(f"fold {fold} epoch {ep + 1}: val F1={vm['F1']:.4f}")
        if vm["F1"] > best:
            best = vm["F1"]
            state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= c["patience"]:
            break

    model.load_state_dict(state)
    model.eval()
    ys = []
    ps = []
    ids = []

    with torch.no_grad():
        for x, m, y, pid in test_loader:
            o = model(x.to(dev), m.to(dev))
            p_final = compute_fused_probs(o)
            ps.extend(p_final.cpu().numpy())
            ys.extend(y.tolist())
            ids.extend(pid)

    os.makedirs(out, exist_ok=True)
    torch.save(
        {"model_state": state, "input_dim": input_dim},
        f"{out}/fold_{fold}.pt"
    )
    pd.DataFrame(hist).to_csv(
        f"{out}/fold_{fold}_history.csv", index=False
    )
    return np.array(ys), np.array(ps), ids
