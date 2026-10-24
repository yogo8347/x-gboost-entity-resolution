"""Per-country composition of two submissions.

usage: python src/compose_country.py BASE_DIR OVERRIDE_DIR COUNTRIES OUT_DIR
Rows whose S1 belongs to one of COUNTRIES (comma list, e.g. France) are taken from OVERRIDE_DIR, all others from BASE_DIR.
Both TSVs (matching_results, candidate_pairs) are composed the same way. Blocking keys are country-prefixed, so an
S2/S3 record can only be matched within its own country and the composed file still assigns each record at most once.
"""
import os, sys
import polars as pl

base, over, countries, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
ctry = pl.read_parquet("data/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
sel = set(ctry.filter(pl.col("country").is_in(countries))["source1_entity_id"].to_list())
os.makedirs(out, exist_ok=True)
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    rd = lambda d: pl.read_csv(f"{d}/{f}", separator="\t", infer_schema_length=0)
    b, o = rd(base), rd(over)
    col = b.columns[0]
    keep = b.filter(~pl.col(col).is_in(sel))
    new = o.filter(pl.col(col).is_in(sel))
    res = pl.concat([keep, new])
    assert res.height == b.height == o.height, (res.height, b.height, o.height)
    res.write_csv(f"{out}/{f}", separator="\t")
    print(f, "rows", res.height, "from override", new.height)
# sanity: no S2/S3 record assigned to two S1s
m = pl.read_csv(f"{out}/matching_results.tsv", separator="\t", infer_schema_length=0)
ids = m["matched_entity_ids"].drop_nulls().str.split(",").explode()
ids = ids.filter(ids != "")
assert ids.n_unique() == ids.len(), "record assigned twice"
print("OK unique assignments", ids.len())
