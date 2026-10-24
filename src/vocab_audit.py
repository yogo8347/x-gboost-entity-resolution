"""Full vocabulary audit (train + test, all sources, names and addresses, per country). Writes data/vocab_audit.txt.

Sections
  1. name words over-represented in S2/S3 vs S1 (injected filler, garbled legal forms, transliterations, honorifics)
  2. address words over-represented in S2/S3 vs S1 (un-normalised abbreviations / spellings)
  3. France (test, unlabelled): words that are EXTRA in a record vs its best S1, confident band vs borderline band
     (candidates for French decoy words and French filler words)
  4. coverage: share of S2/S3 name-word occurrences that never occur in any S1 name
"""
import sys
import polars as pl

sys.path.insert(0, "src")
from normalize import LEGAL
from features import LEGALISH
import re

OUT = open("data/vocab_audit.txt", "w", encoding="utf-8")
def w(*a):
    print(*a); print(*a, file=OUT); OUT.flush()


def tok_counts(split, k, col, country=None):
    lf = pl.scan_parquet(f"data/{split}_s{k}.parquet").select(col, "country")
    if country:
        lf = lf.filter(pl.col("country") == country)
    return (lf.with_columns(pl.col(col).str.split(" ")).explode(col).filter(pl.col(col).str.len_chars() >= 2)
              .group_by(col).agg(pl.len().alias("n")).collect(engine="streaming").rename({col: "w"}))


cleaned = re.compile(LEGALISH)
for col, title in (("name", "NAME"), ("addr", "ADDRESS")):
    for country in ("US", "India", "France"):
        splits = ("test",) if country == "France" else ("train", "test")
        s1 = pl.concat([tok_counts(sp, 1, col, country) for sp in splits]).group_by("w").agg(pl.col("n").sum().alias("n_s1"))
        qq = pl.concat([tok_counts(sp, k, col, country) for sp in splits for k in (2, 3)]).group_by("w").agg(pl.col("n").sum().alias("n_q"))
        n1, nq = s1["n_s1"].sum(), qq["n_q"].sum()
        d = qq.join(s1, on="w", how="left").fill_null(0).with_columns(
            (((pl.col("n_q") / nq) + 1e-7) / ((pl.col("n_s1") / n1) + 1e-7)).alias("ratio"))
        d = d.with_columns(pl.col("w").map_elements(lambda x: "legal-list" if x in LEGAL else ("cleaner" if cleaned.fullmatch(x) else ""),
                                                    return_dtype=pl.Utf8).alias("handled"))
        top = d.filter((pl.col("n_q") >= 1500) & (pl.col("ratio") >= 3)).sort("n_q", descending=True).head(60)
        w(f"\n=== {title} words over-represented in S2/S3 vs S1 — {country} ({'test' if country == 'France' else 'train+test'}) ===")
        w("word | S2/S3 count | S1 count | ratio | already handled")
        for r in top.iter_rows(named=True):
            w(f"  {r['w']:16s} {r['n_q']:9d} {r['n_s1']:8d} {r['ratio']:9.1f}  {r['handled']}")
        if col == "name":
            unk = d.filter(pl.col("n_s1") == 0)["n_q"].sum()
            w(f"  coverage: {unk / nq:.3%} of S2/S3 name-word occurrences never appear in any {country} S1 name")

# 3. France: extra words vs best S1, confident vs borderline stage-1 band
S = pl.scan_parquet("data/frpatch/test_scored.parquet").select("qid", "rid", "p")
top = S.sort(["qid", "p"], descending=[False, True]).group_by("qid", maintain_order=True).first().collect(engine="streaming")
s1 = pl.read_parquet("data/frpatch/test_s1.parquet", columns=["name", "country"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
q = pl.concat([pl.read_parquet(f"data/frpatch/test_s{k}.parquet", columns=["name"]) for k in (2, 3)]).with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
fr = top.join(s1.filter(pl.col("country") == "France"), on="rid").join(q.rename({"name": "q_name"}), on="qid")
fr = fr.with_columns(pl.col("q_name").str.split(" ").list.set_difference(pl.col("name").str.split(" ")).alias("extra"))
bands = {"confident (p>0.99)": fr.filter(pl.col("p") > 0.99), "borderline (0.3<p<0.9)": fr.filter((pl.col("p") > 0.3) & (pl.col("p") < 0.9))}
cnt = {}
for b, d in bands.items():
    c = d.select(pl.col("extra")).explode("extra").filter(pl.col("extra").str.len_chars() >= 2).group_by("extra").agg(pl.len().alias("n"))
    cnt[b] = c.with_columns((pl.col("n") / d.height).alias("rate_" + b.split()[0]))
m = cnt["confident (p>0.99)"].join(cnt["borderline (0.3<p<0.9)"], on="extra", how="full", coalesce=True).fill_null(0)
m = m.with_columns(((pl.col("rate_borderline") + 1e-5) / (pl.col("rate_confident") + 1e-5)).alias("borderline_lift"))
w("\n=== FRANCE: extra words (record has, best S1 lacks) — borderline vs confident band ===")
w(f"records: confident {bands['confident (p>0.99)'].height}, borderline {bands['borderline (0.3<p<0.9)'].height}")
w("words much more typical of BORDERLINE France pairs (French decoy-word candidates):")
for r in m.filter(pl.col("n_right") >= 30 if "n_right" in m.columns else pl.lit(True)).sort("borderline_lift", descending=True).head(30).iter_rows(named=True):
    w(f"  {r['extra']:16s} borderline-rate {r['rate_borderline']:.4f}  confident-rate {r['rate_confident']:.4f}  lift {r['borderline_lift']:.1f}")
w("words typical of CONFIDENT France pairs (French filler / alias candidates):")
for r in m.filter(pl.col("rate_confident") > 0.001).sort("borderline_lift").head(20).iter_rows(named=True):
    w(f"  {r['extra']:16s} borderline-rate {r['rate_borderline']:.4f}  confident-rate {r['rate_confident']:.4f}  lift {r['borderline_lift']:.2f}")
OUT.close()
