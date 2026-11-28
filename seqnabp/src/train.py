import os
import copy

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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

    if len(loader.dataset) == 0:
        raise ValueError("An empty data loader was supplied to epoch().")

    return total / len(loader.dataset), metrics(ys, ps)


def _make_cluster_grouped_inner_split(tr, metadata, fold, c):
   
    tr_arr = np.asarray(tr)

    missing = [
        str(pid) for pid in tr_arr
        if "cluster_id" not in metadata[pid]
        or metadata[pid]["cluster_id"] is None
        or str(metadata[pid]["cluster_id"]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "Cluster-grouped inner validation requires a non-empty "
            f"cluster_id for every outer-training protein. Missing examples: "
            f"{missing[:10]}"
        )

    tr_labels = np.asarray(
        [int(metadata[pid]["label"]) for pid in tr_arr], dtype=np.int64
    )
    tr_groups = np.asarray(
        [str(metadata[pid]["cluster_id"]) for pid in tr_arr]
    )

    if len(np.unique(tr_groups)) < 8:
        raise ValueError(
            f"Fold {fold}: at least 8 distinct clusters are required for "
            "8-fold inner validation, but only "
            f"{len(np.unique(tr_groups))} were found."
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=8,
        shuffle=True,
        random_state=int(c["seed"]) + int(fold)
    )
    splits = list(
        inner_cv.split(np.zeros(len(tr_arr)), tr_labels, tr_groups)
    )

    inner_train_idx, stop_idx = splits[0]
    tr_inner = tr_arr[inner_train_idx]
    stop_inner = tr_arr[stop_idx]

    train_groups = set(tr_groups[inner_train_idx])
    stop_groups = set(tr_groups[stop_idx])
    shared_groups = train_groups.intersection(stop_groups)
    if shared_groups:
        raise RuntimeError(
            f"Fold {fold}: inner training/validation cluster overlap "
            f"detected ({len(shared_groups)} shared clusters)."
        )

    all_classes = set(np.unique(tr_labels).tolist())
    train_classes = set(tr_labels[inner_train_idx].tolist())
    val_classes = set(tr_labels[stop_idx].tolist())

    if train_classes != all_classes:
        raise ValueError(
            f"Fold {fold}: inner training split is missing class(es): "
            f"{sorted(all_classes - train_classes)}. Consider revising "
            "the cluster assignments or inner split strategy."
        )
    if val_classes != all_classes:
        raise ValueError(
            f"Fold {fold}: inner validation partition is missing class(es): "
            f"{sorted(all_classes - val_classes)}. The selected "
            "StratifiedGroupKFold partition is unsuitable for calculating "
            "the intended validation metrics. Inspect cluster-level class "
            "distribution before training."
        )

    print(
        f"Fold {fold}: inner training={len(tr_inner)} proteins, "
        f"validation={len(stop_inner)} proteins, "
        f"training clusters={len(train_groups)}, "
        f"validation clusters={len(stop_groups)}, shared clusters=0"
    )
    print(
        f"Fold {fold}: inner validation fraction="
        f"{len(stop_inner) / len(tr_arr):.3%}"
    )

    return tr_inner, stop_inner


def train_fold(safetensors_path, metadata, tr, va, fold, c, out,
               input_dim):
    seed_all(c["seed"] + fold)
    dev = device()

    tr_inner, stop_inner = _make_cluster_grouped_inner_split(
        tr, metadata, fold, c
    )

    outer_train_groups = {
        str(metadata[pid]["cluster_id"]) for pid in tr
    }
    outer_test_groups = {
        str(metadata[pid]["cluster_id"]) for pid in va
    }
    if outer_train_groups.intersection(outer_test_groups):
        raise RuntimeError(
            f"Fold {fold}: outer training/test cluster overlap detected."
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

    if state is None:
        raise RuntimeError(
            f"Fold {fold}: no valid model checkpoint was selected."
        )

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
