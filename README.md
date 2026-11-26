# SeqNABP-MS

**Multiscale Sequence Modeling and Confidence-Aware Classification of Nucleic Acid-Binding Proteins**

SeqNABP-MS is a sequence-based deep learning framework for identifying nucleic acid-binding proteins (NABPs) and resolving their fine-grained binding specificities across four functional classes:
* **non-NABP**: Non-nucleic acid-binding proteins
* **RBP**: RNA-binding proteins
* **DSB**: Double-stranded DNA-binding proteins
* **SSB**: Single-stranded DNA-binding proteins

---

## Repository Structure

```text
seqnabp/
├── config.json                              # Model & training configuration
├── requirements.txt                         # Python dependencies
├── data/
│   └── proteins.csv                         # Input sequences, labels
└── src/
    ├── utils.py                             # Label maps, seeding, device helpers
    ├── embeddings.py                        # ESM-2 extraction to .safetensors
    ├── data_utils.py                        # Lazy SafeTensors dataset & collation
    ├── model.py                             # SeqNABP-MS architecture definition
    ├── train.py                             # Fold training & probability fusion
    ├── evaluate.py                          # Multi-class evaluation metrics
    ├── cv.py                                # 5-fold cluster-grouped CV runner
    ├── export_paper_tables.py               # Per-class & hierarchical summary tables
    ├── analysis1_attention_map.py           # Residue-level attention profiles
    ├── analysis2_calibration_uncertainty.py # Temperature scaling & risk-coverage
    ├── analysis3_error_breakdown.py         # Confusion matrix, ROC/PR, length bins
    └── analysis4_homology_generalization.py # Cluster & sequence identity evaluation
```

---

## Installation

1. **Create and activate a Conda environment (Python 3.10–3.12):**
   ```bash
   conda create -n seqnabp python=3.12 -y
   conda activate seqnabp
   ```

2. **Install PyTorch with CUDA support** (adjust CUDA version for your GPU):
   ```bash
   pip install torch --index-url https://pytorch.org
   ```

3. **Install remaining dependencies (including `safetensors` and `mamba-ssm`):**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `mamba-ssm` and `causal-conv1d` require a compatible NVIDIA GPU and CUDA toolkit.)*

---

## Usage

### Step 1: Prepare Input Data & Checkpoint
Place your dataset at `data/proteins.csv` with the following columns:
* `protein_id`: Unique sequence identifier
* `sequence`: Amino acid sequence (length 40–1,500 aa)
* `label`: One of `nonNABP`, `RBP`, `DSB`, or `SSB`
* `cluster_id`: Homology cluster identifier (e.g., from CD-HIT at 40% identity)

Download the pretrained ESM-2 checkpoint (`esm2_t36_3B_UR50D.pt`) and set its path in `config.json`.

### Step 2: Extract ESM-2 Residue Embeddings
Extract and store per-residue embeddings in `.safetensors` format alongside a lightweight JSON metadata index:
```bash
python -m src.embeddings --config config.json
```

### Step 3: Run 5-Fold Cluster-Grouped Cross-Validation
Train and evaluate SeqNABP-MS across all five homology-controlled folds:
```bash
python -m src.cv --config config.json
```
Outputs (`fold_1.pt`–`fold_5.pt`, `fold_metrics.csv`, `oof_predictions.csv`, and `cv_summary.json`) are saved in `results/`.

### Step 4: Downstream Calibration, Error, and Interpretability Analyses
```bash
# Export 5-fold Mean ± SD class-wise and hierarchical tables
python -m src.export_paper_tables --config config.json

# Run post-hoc temperature scaling & risk-coverage analysis
python -m src.analysis2_calibration_uncertainty --config config.json

# Generate confusion matrix, ROC/PR curves, and sequence-length breakdown
python -m src.analysis3_error_breakdown --config config.json

# Generate residue-level attention maps on query FASTA files
python -m src.analysis1_attention_map --config config.json --fold 1
```

---

## Citation

If you use SeqNABP-MS or this codebase in your research, please cite our work:

```bibtex
Citation will be available once accept.
```
