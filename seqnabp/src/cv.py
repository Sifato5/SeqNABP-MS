import os
import json
import argparse

import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.model_selection import StratifiedGroupKFold

from .utils import config, seed_all, NAMES
from .train import train_fold
from .evaluate import metrics


def _validate_metadata(metadata):
    if not metadata:
        raise ValueError("The metadata file contains no protein records.")

    missing_label = [
        pid for pid, item in metadata.items()
        if "label" not in item
    ]
    if missing_label:
        raise ValueError(
            "Missing labels for proteins; examples: "
            f"{missing_label[:10]}"
        )

    missing_cluster = [
        pid for pid, item in metadata.items()
        if "cluster_id" not in item
        or item["cluster_id"] is None
        or str(item["cluster_id"]).strip() == ""
    ]
    if missing_cluster:
        raise ValueError(
            "Cluster IDs are mandatory for cluster-grouped cross-validation. "
            f"Missing/empty IDs for {len(missing_cluster)} proteins; "
            f"examples: {missing_cluster[:10]}. Add cluster_id to the "
            "metadata JSON (for example, from the CD-HIT .clstr mapping) "
            "and regenerate the metadata before running CV."
        )

    labels = np.asarray(
        [int(item["label"]) for item in metadata.values()],
        dtype=np.int64
    )
    observed = set(np.unique(labels).tolist())
    expected = {0, 1, 2, 3}
    if observed != expected:
        raise ValueError(
            "This training pipeline expects all four class labels "
            f"{sorted(expected)}; observed labels were {sorted(observed)}."
        )

    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    a = p.parse_args()

    c = config(a.config)
    seed_all(c["seed"])

    n_splits = int(c.get("n_splits", 5))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if c.get("use_cluster_groups", True) is not True:
        raise ValueError(
            "use_cluster_groups must be true. Random stratified CV is "
            "disabled because it does not prevent homology leakage."
        )

    sf_path = c["embedding_file"]
    if sf_path.endswith(".pt"):
        sf_path = sf_path.rsplit(".", 1)[0] + ".safetensors"
    if not os.path.isfile(sf_path):
        raise FileNotFoundError(f"Embedding file not found: {sf_path}")

    meta_path = sf_path.replace(".safetensors", "_metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    y = _validate_metadata(metadata)

    with safe_open(sf_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No embeddings were found in {sf_path}.")
        first_key = keys[0]
        input_dim = f.get_tensor(first_key).shape[-1]

    pids = np.array(list(metadata.keys()))
    y = np.array([int(metadata[pid]["label"]) for pid in pids])
    groups = np.array(
        [str(metadata[pid]["cluster_id"]) for pid in pids]
    )

    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"At least {n_splits} distinct clusters are required for "
            f"{n_splits}-fold outer CV; found {len(np.unique(groups))}."
        )

    out = c["output_dir"]
    os.makedirs(out, exist_ok=True)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=c["seed"]
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))

    all_classes = set(np.unique(y).tolist())
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        train_groups = set(groups[tr_idx])
        test_groups = set(groups[va_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(
                f"Outer fold {fold}: {len(overlap)} cluster(s) overlap "
                "between training and test sets."
            )

        train_classes = set(y[tr_idx].tolist())
        test_classes = set(y[va_idx].tolist())
        if train_classes != all_classes:
            raise ValueError(
                f"Outer fold {fold} training partition is missing "
                f"class(es): {sorted(all_classes - train_classes)}."
            )
        if test_classes != all_classes:
            raise ValueError(
                f"Outer fold {fold} test partition is missing "
                f"class(es): {sorted(all_classes - test_classes)}. "
                "Check the number and class composition of CD-HIT clusters."
            )

        print(
            f"Outer fold {fold}: train={len(tr_idx)}, test={len(va_idx)}, "
            f"train clusters={len(train_groups)}, "
            f"test clusters={len(test_groups)}, shared clusters=0"
        )

    print(
        f"Running {n_splits}-fold cluster-grouped stratified CV "
        f"on {len(pids)} proteins and {len(np.unique(groups))} clusters."
    )

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
    missing_metrics = [k for k in metric_cols if k not in df.columns]
    if missing_metrics:
        raise KeyError(
            f"Expected metric columns were not returned: {missing_metrics}"
        )

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
            f.write(
                f'{k}: {v["mean"]:.4f} +/- {v["std"]:.4f}\n'
            )

    print(f"\n{n_splits}-fold mean +/- sample SD:")
    disp = pd.DataFrame({
        k: [f'{v["mean"]:.4f} +/- {v["std"]:.4f}']
        for k, v in summary.items()
    })
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
