# SeqNABP-MS

**Multiscale Sequence Modeling and Confidence-Aware Classification of Nucleic Acid-Binding Proteins**

SeqNABP-MS is a sequence-based deep learning framework for identifying nucleic acid-binding proteins (NABPs) and resolving their binding specificities across four functional classes:
* **non-NABP**: Non-nucleic acid-binding proteins
* **RBP**: RNA-binding proteins
* **DSB**: Double-stranded DNA-binding proteins
* **SSB**: Single-stranded DNA-binding proteins

---

## Model Overview
* **PLM Residue Representations**: Extracts per-residue embeddings from ESM-2 (`esm2_t36_3B_UR50D`, Layer 36).
* **Hybrid Sequence Encoder**: Combines multiscale residual dilated convolutions, selective state-space modeling (Mamba), and multi-head self-attention via adaptive gated fusion.
* **Attentive Statistical Pooling**: Aggregates residue features using attention-weighted mean, standard deviation, global mean, and global max.
* **Unified Multi-Task & Pathway Fusion**: Jointly optimizes direct four-class logits and hierarchical conditional pathways (`NABP -> DBP/RBP -> SSB/DSB`).

---

## Repository Structure

```text
seqnabp/
├── config.json          # Model and training hyperparameters
├── requirements.txt     # Python dependencies
├── data/
│   └── proteins.csv     # Input sequences, labels, and cluster IDs
└── src/
    ├── utils.py         # Label mappings, seeding, and device utilities
    ├── embeddings.py    # ESM-2 residue embedding extraction (.safetensors)
    ├── data_utils.py    # Lazy SafeTensors dataset and batch collation
    ├── model.py         # SeqNABPX network architecture
    ├── train.py         # Single-fold training and probability fusion
    ├── evaluate.py      # Multi-class evaluation metrics
    └── cv.py            # 5-fold cluster-grouped cross-validation pipeline
```

---

## Installation

1. **Create and activate a Conda environment (Python 3.10–3.12):**
   ```bash
   conda create -n seqnabp python=3.12 -y
   conda activate seqnabp
   ```

2. **Install PyTorch with CUDA support** (match your local CUDA version):
   ```bash
   pip install torch --index-url https://pytorch.org
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `mamba-ssm` and `causal-conv1d` require a compatible NVIDIA GPU and CUDA toolkit.)*

---

## Usage

### 1. Prepare Input Data and ESM-2 Checkpoint
Place your dataset at `data/proteins.csv` with the following columns:
* `protein_id`: Unique sequence identifier
* `sequence`: Amino acid sequence (length 40–1,500 aa)
* `label`: One of `nonNABP`, `RBP`, `DSB`, or `SSB`
* `cluster_id`: Homology cluster identifier (e.g., CD-HIT at 40% sequence identity)

Download the pretrained ESM-2 checkpoint (`esm2_t36_3B_UR50D.pt`) and specify its path in `config.json`.

### 2. Extract ESM-2 Residue Embeddings
Generate per-residue embeddings in `.safetensors` format alongside the metadata sidecar JSON:
```bash
python -m src.embeddings --config config.json
```

### 3. Run 5-Fold Cluster-Grouped Cross-Validation
Train and evaluate SeqNABP-MS across all five homology-controlled folds:
```bash
python -m src.cv --config config.json
```
Checkpoints (`fold_1.pt`–`fold_5.pt`), fold-wise metrics (`fold_metrics.csv`), out-of-fold predictions (`oof_predictions.csv`), and summary statistics (`cv_summary.json`) will be saved in `results/`.

---

## Citation

If you use SeqNABP-MS in your research, please cite:

```bibtex
Citation will available once accept.
```
