"""Robust gene-position fetch: download a static Ensembl GTF once (no live API/XML query,
which failed via pybiomart/biomart with a malformed-XML error) and parse chromosome/start/end
per gene symbol. Run on the LOGIN NODE (needs internet; compute nodes don't have it).
"""
import gzip
import urllib.request
from pathlib import Path

import pandas as pd

GTF_URL = "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz"
LOCAL_GZ = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build/grch38.gtf.gz")
OUT_PARQUET = Path("/shared/projects/tp_2630_ubordeaux_neuromics_184418/projects/C10/lederer/gbm_space_proj/scratch_build/grch38_gene_positions.parquet")

MAIN_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def download():
    if LOCAL_GZ.exists() and LOCAL_GZ.stat().st_size > 10_000_000:
        print(f"Already downloaded: {LOCAL_GZ} ({LOCAL_GZ.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"Downloading {GTF_URL} ...")
    urllib.request.urlretrieve(GTF_URL, LOCAL_GZ)
    print(f"Downloaded {LOCAL_GZ.stat().st_size / 1e6:.0f} MB")


def parse_gene_name(attr_field: str) -> str | None:
    for part in attr_field.split(";"):
        part = part.strip()
        if part.startswith("gene_name"):
            return part.split(" ")[1].strip('"')
    return None


def parse():
    rows = []
    with gzip.open(LOCAL_GZ, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom = fields[0]
            if chrom not in MAIN_CHROMS:
                continue
            gene_name = parse_gene_name(fields[8])
            if gene_name is None:
                continue
            rows.append((gene_name, chrom, int(fields[3]), int(fields[4])))

    df = pd.DataFrame(rows, columns=["gene_name", "chromosome", "start", "end"])
    print(f"Parsed {len(df):,} gene records (main chromosomes only) before dedup")
    # A few symbols appear more than once (paralogs/readthroughs on different loci) —
    # keep the first occurrence, consistent with how var_names_make_unique handles dups.
    df = df.drop_duplicates(subset="gene_name", keep="first").set_index("gene_name")
    df["chromosome"] = "chr" + df["chromosome"]
    df.to_parquet(OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET} ({len(df):,} unique gene symbols)")
    print(df.head())


if __name__ == "__main__":
    download()
    parse()
