import os
import json
import argparse
import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from .utils import config, seed_all, NAMES
from .train import train_fold
from .evaluate import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    a = p.parse_args()

    c = config(a.config)
    seed_all(c["seed"])

    sf_path = c["embedding_file"]
    if sf_path.endswith(".pt"):
        sf_path = sf_path.rsplit(".", 1)[0] + ".safetensors"
    meta_path = sf_path.replace(".safetensors", "_metadata.json")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    with safe_open(sf_path, framework="pt", device="cpu") as f:
        first_key = next(iter(f.keys()))
        input_dim = f.get_tensor(first_key).shape[-1]

    pids = np.array(list(metadata.keys()))
    y = np.array([metadata[pid]["label"] for pid in pids])
    out = c["output_dir"]
    os.makedirs(out, exist_ok=True)

    has_clusters = all("cluster_id" in metadata[pid] for pid in pids)
    grouped = c.get("use_cluster_groups", True) and has_clusters

    if grouped:
        g = np.array([metadata[pid]["cluster_id"] for pid in pids])
        sp = StratifiedGroupKFold(
            c["n_splits"], shuffle=True, random_state=c["seed"]
        )
        splits = list(sp.split(np.zeros(len(y)), y, g))
        print("Cluster-grouped stratified CV")
    else:
        sp = StratifiedKFold(
            c["n_splits"], shuffle=True, random_state=c["seed"]
        )
        splits = list(sp.split(np.zeros(len(y)), y))
        print("WARNING: random stratified CV; leakage not prevented.")

    rows = []
    preds = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        tr_pids = pids[tr_idx]
        va_pids = pids[va_idx]

        yt, pr, ids = train_fold(
            sf_path, metadata, tr_pids, va_pids, fold, c, out, input_dim
        )
        m = metrics(yt, pr)
        rows.append({"fold": fold, **m})

        for pid, t, q in zip(ids, yt, pr):
            preds.append({
                "fold": fold,
                "protein_id": pid,
                "true_label": NAMES[int(t)],
                "predicted_label": NAMES[int(q.argmax())],
                **{f"prob_{NAMES[i]}": float(q[i]) for i in range(4)}
            })
        print(f"Fold {fold} metrics:", m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "fold_metrics.csv"), index=False)
    pd.DataFrame(preds).to_csv(
        os.path.join(out, "oof_predictions.csv"), index=False
    )

    metric_cols = ["ACC", "REC", "PRE", "F1", "AUC", "AUPRC"]
    summary = {
        k: {
            "mean": float(df[k].mean()),
            "std": float(df[k].std(ddof=1))
        }
        for k in metric_cols
    }

    with open(os.path.join(out, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out, "cv_summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f'{k}: {v["mean"]:.4f} +/- {v["std"]:.4f}\n')

    print("\n5-fold mean +/- sample SD:")
    disp = pd.DataFrame({
        k: [f'{v["mean"]:.4f} +/- {v["std"]:.4f}']
        for k, v in summary.items()
    })
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
