"""CE v2 (v12): training pairs for a stronger cross-encoder, taken from the stage-1 TRAINING records only.

v10's cross-encoder was trained on ~195k held-out-fold pairs per fold. Here it is trained on best-candidate pairs of the
records stage 1 trained on (select_training_records: true S1 in the training fold AND no held-out S1 among the
candidates), so it never sees a held-out S1 entity, and ONE model then scores every validation pair and every test pair
(validation and test scores come from the same model -> no fold-mixing drift).

Hardness selection: stage-1 in-sample probability of each record's best candidate in [LO, HI] (in-sample scores are
over-confident, so the band is wider than v10's [0.02, 0.995]), plus the runner-up when it is a true match.
  python src/ce2_prep.py   -> cloud_package2/train_pairs.parquet (qid, rid, qt, st, label)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import lightgbm as lgb
from cross_encoder import _raw, _text

LO, HI, CAP = 0.002, 0.9995, 900_000
m = lgb.Booster(model_file="data/model.txt"); names = m.feature_name()
F = pl.read_parquet("data/train_feats.parquet", columns=["qid", "rid", "label", *names])
p = np.zeros(F.height, np.float32)
for i in range(0, F.height, 1_000_000):
    p[i:i + 1_000_000] = m.predict(F[i:i + 1_000_000].select(names).to_numpy().astype(np.float32), num_threads=8)
    print("scored", min(i + 1_000_000, F.height), flush=True)
F = F.select("qid", "rid", "label").with_columns(pl.Series("p", p))
F = F.with_columns(pl.col("p").rank("ordinal", descending=True).over("qid").alias("r"))
best = F.filter((pl.col("r") == 1) & pl.col("p").is_between(LO, HI))
second = F.filter((pl.col("r") == 2) & (pl.col("label") == 1))
print("best in band", best.height, "pos rate", round(best["label"].mean(), 3), "| true runner-ups", second.height, flush=True)
t = pl.concat([best, second])
if t.height > CAP:
    t = t.sample(CAP, seed=3)
q = pl.concat([_raw("train", 2), _raw("train", 3)]).with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
q = q.join(t.select("qid").unique(), on="qid", how="semi")
s1 = _raw("train", 1).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).join(t.select("rid").unique(), on="rid", how="semi")
q = q.with_columns(pl.Series("qt", _text(q))).select("qid", "qt")
s1 = s1.with_columns(pl.Series("st", _text(s1))).select("rid", "st")
t = t.join(q, on="qid").join(s1, on="rid").select("qid", "rid", "qt", "st", pl.col("label").cast(pl.Int8), "p")
os.makedirs("cloud_package2", exist_ok=True)
t.write_parquet("cloud_package2/train_pairs.parquet")
print("wrote cloud_package2/train_pairs.parquet", t.height, "pos rate", round(t["label"].mean(), 3))
print(t.sample(5, seed=1).select("qt", "st", "label"))
