import argparse, os, pandas as pd, torch, esm
import json
from tqdm import tqdm
from safetensors.numpy import save_file
import numpy as np
from .utils import config, seed_all
from .data_utils import label_id

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config.json")
    p.add_argument("--input"); p.add_argument("--output"); p.add_argument("--checkpoint")
    a = p.parse_args(); c = config(a.config); seed_all(c["seed"])
    inp = a.input or c["data_csv"]; out = a.output or c["embedding_file"]; ck = a.checkpoint or c["esm_checkpoint"]
    
    if out.endswith(".pt"):
        out = out.rsplit(".", 1)[0] + ".safetensors"
        
    if not os.path.isfile(ck): raise FileNotFoundError(f"Checkpoint not found: {ck}")
    df = pd.read_csv(inp)
    if not {"sequence", "label"}.issubset(df.columns): raise ValueError("CSV needs sequence,label columns")
    
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(ck)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval().to(dev)
    convert = alphabet.get_batch_converter()
    
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    
    tensors_to_save = {}
    metadata = {}
    dim_example = None
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        seq = "".join(str(row.sequence).split()).upper()
        if not seq: raise ValueError(f"Empty sequence at row {i}")
        if len(seq) > c["max_length"]: raise ValueError(f"Sequence {i} length {len(seq)} > max_length")
        
        pid = str(row["protein_id"]) if "protein_id" in df.columns else str(i)
        _, _, tok = convert([(pid, seq)]); tok = tok.to(dev)
        
        with torch.no_grad():
            rep = model(tok, repr_layers=[c["embedding_layer"]])["representations"][c["embedding_layer"]][0, 1:len(seq)+1].float().cpu().numpy()
        
        if dim_example is None:
            dim_example = rep.shape[-1]
            
        tensors_to_save[f"emb_{pid}"] = rep
        
        meta_entry = {"label": int(label_id(row.label))}
        if "cluster_id" in df.columns and pd.notna(row.cluster_id): 
            meta_entry["cluster_id"] = str(row.cluster_id)
        metadata[pid] = meta_entry

    save_file(tensors_to_save, out)
    
    meta_out = out.replace(".safetensors", "_metadata.json")
    with open(meta_out, "w") as f:
        json.dump(metadata, f)
        
    print(f"Saved records to {out}; embedding dimension {dim_example}")

if __name__ == "__main__": main()