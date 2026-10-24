"""Normalise every record of a split once and cache to parquet (data/<split>_sN.parquet)."""
import sys, os
import polars as pl
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from normalize import norm_name, norm_addr

def _work(rows):
    out = []
    for eid, name, addr, ctry in rows:
        full, core, dba = norm_name(name)
        a, nums, words, _ = norm_addr(addr, ctry)
        out.append((eid, ctry, full, core, dba, a, " ".join(nums), " ".join(words)))
    return out

def run(data_dir, split, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for k in (1, 2, 3):
        dst = f"{out_dir}/{split}_s{k}.parquet"
        if os.path.exists(dst):
            continue
        df = pl.read_csv(f"{data_dir}/{split}/{split}_source{k}.tsv", separator="\t",
                         quote_char=None, infer_schema=False)
        rows = list(zip(df["entity_id"], df["business_name"], df["business_address"], df["country"]))
        del df
        chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
        with Pool(7) as p:
            res = [r for c in p.imap(_work, chunks, chunksize=4) for r in c]
        del rows, chunks
        cols = ["entity_id", "country", "name", "core", "dba", "addr", "nums", "awords"]
        pl.DataFrame(res, schema=cols, orient="row").write_parquet(dst)
        print("wrote", dst, len(res), flush=True)

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dataset", sys.argv[2] if len(sys.argv) > 2 else "train")
