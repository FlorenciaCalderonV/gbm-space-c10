"""Fetch GRCh38 gene genomic positions from Ensembl biomart (login node, has internet)
and cache as parquet so infercnvpy can run offline on the compute node.
Keyed by HGNC symbol to match the data's var_names.
"""
from pathlib import Path
import pandas as pd

WORK = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build")

try:
    from pybiomart import Server
    server = Server(host="http://www.ensembl.org")
    mart = server["ENSEMBL_MART_ENSEMBL"]["hsapiens_gene_ensembl"]
    df = mart.query(attributes=[
        "external_gene_name", "chromosome_name", "start_position", "end_position",
    ])
    df.columns = ["gene_name", "chromosome", "start", "end"]
    # keep standard chromosomes
    keep_chr = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    df = df[df["chromosome"].isin(keep_chr)].copy()
    df["chromosome"] = "chr" + df["chromosome"].astype(str)
    df = df.dropna(subset=["start", "end"]).drop_duplicates("gene_name")
    df = df.set_index("gene_name")
    df.to_parquet(WORK / "grch38_gene_positions.parquet")
    print(f"Saved {len(df)} gene positions to grch38_gene_positions.parquet")
    print(df.head())
    print("chromosomes:", sorted(df["chromosome"].unique()))
except Exception as e:
    import traceback; traceback.print_exc()
    print("BIOMART FAILED:", e)
