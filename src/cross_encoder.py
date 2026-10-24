"""Transformer cross-encoder for the best candidate pair (Ditto-style), as an extra stage-2 input.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters, Apache-2.0), fine-tuned to read
"name | address [SEP] name | address" (anyascii-transliterated) and output P(same business).
Only UNCERTAIN best pairs are scored (stage-1 p1 in [LO, HI]); the rest keep ce = NaN (the tree model handles it).
Leak-free: 2 folds keyed by S1 entity on the held-out fold (train on one half, predict the other);
test gets the average of both fold models.

  python src/cross_encoder.py prep            # text pairs for validation + test uncertain rows
  python src/cross_encoder.py train <fold>    # fine-tune on the other fold, predict this fold + test
  python src/cross_encoder.py merge           # -> data/ce_val.parquet, data/ce_test.parquet
"""
import os, sys, time
import numpy as np
import polars as pl

LO, HI = 0.02, 0.995
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAXLEN, BS_TRAIN, BS_PRED = 96, 64, 512
OUT = "data/ce"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _raw(split, k):
    return pl.read_csv(f"dataset/{split}/{split}_source{k}.tsv", separator="\t", quote_char=None, infer_schema=False,
                       columns=["business_name", "business_address"])


def _text(df):
    from anyascii import anyascii
    n = df["business_name"].fill_null("").to_list(); a = df["business_address"].fill_null("").to_list()
    return [anyascii(x + " | " + y)[:300] for x, y in zip(n, a)]


def prep():
    for split, table in (("train", "data/stage2v8_train.parquet"), ("test", "data/frpatch/stage2v8_test.parquet")):
        cols = ["qid", "rid", "p1"] + (["label", "s1_id"] if split == "train" else [])
        t = pl.read_parquet(table, columns=cols).filter((pl.col("p1") >= LO) & (pl.col("p1") <= HI))
        q = pl.concat([_raw(split, 2), _raw(split, 3)]).with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32))
        q = q.join(t.select("qid"), on="qid", how="semi")
        s1 = _raw(split, 1).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)).join(t.select("rid").unique(), on="rid", how="semi")
        q = q.with_columns(pl.Series("qt", _text(q))).select("qid", "qt")
        s1 = s1.with_columns(pl.Series("st", _text(s1))).select("rid", "st")
        t = t.join(q, on="qid").join(s1, on="rid")
        if split == "train":
            t = t.with_columns((pl.col("s1_id").hash(seed=77) % 2).alias("fold"))
        t.write_parquet(f"{OUT}/{split}_pairs.parquet")
        log(split, "uncertain best pairs:", t.height)


def _batches(qt, st, bs, tok):
    for i in range(0, len(qt), bs):
        yield i, tok(qt[i:i + bs], st[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")


def predict(model, tok, qt, st, dev):
    import torch
    model.eval(); out = np.zeros(len(qt), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i, b in _batches(qt, st, BS_PRED, tok):
            b = {k: v.to(dev) for k, v in b.items()}
            out[i:i + len(b["input_ids"])] = torch.sigmoid(model(**b).logits.float().squeeze(-1)).cpu().numpy()
    return out


def train(fold):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    tr = pl.read_parquet(f"{OUT}/train_pairs.parquet").filter((pl.col("p1") >= LO) & (pl.col("p1") <= HI))
    fit = tr.filter(pl.col("fold") != fold).sample(fraction=1.0, shuffle=True, seed=fold)
    qt, st, y = fit["qt"].to_list(), fit["st"].to_list(), fit["label"].to_numpy().astype(np.float32)
    log(f"fold {fold}: fine-tune on {len(y)} pairs (pos rate {y.mean():.3f}) on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = (len(y) + BS_TRAIN - 1) // BS_TRAIN
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-5, total_steps=steps, pct_start=0.06)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for step, (i, b) in enumerate(_batches(qt, st, BS_TRAIN, tok)):
        b = {k: v.to(dev) for k, v in b.items()}
        yy = torch.tensor(y[i:i + BS_TRAIN], device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = lossf(model(**b).logits.float().squeeze(-1), yy)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0:
            log(f"  step {step}/{steps} loss {loss.item():.4f}")
    model.save_pretrained(f"{OUT}/model_fold{fold}"); tok.save_pretrained(f"{OUT}/model_fold{fold}")
    ho = tr.filter(pl.col("fold") == fold)
    p = predict(model, tok, ho["qt"].to_list(), ho["st"].to_list(), dev)
    from sklearn.metrics import roc_auc_score
    log(f"fold {fold}: held-out AUC {roc_auc_score(ho['label'].to_numpy(), p):.4f}  vs stage-1 p1 AUC {roc_auc_score(ho['label'].to_numpy(), ho['p1'].to_numpy()):.4f}")
    ho.select("qid").with_columns(pl.Series("ce", p)).write_parquet(f"{OUT}/val_fold{fold}.parquet")
    te = pl.read_parquet(f"{OUT}/test_pairs.parquet").filter((pl.col("p1") >= LO) & (pl.col("p1") <= HI))
    pt = predict(model, tok, te["qt"].to_list(), te["st"].to_list(), dev)
    te.select("qid").with_columns(pl.Series("ce", pt)).write_parquet(f"{OUT}/test_fold{fold}.parquet")
    log(f"fold {fold}: test predictions {len(pt)}")


def merge():
    v = pl.concat([pl.read_parquet(f"{OUT}/val_fold{k}.parquet") for k in (0, 1)])
    t0, t1 = pl.read_parquet(f"{OUT}/test_fold0.parquet"), pl.read_parquet(f"{OUT}/test_fold1.parquet")
    t = t0.join(t1, on="qid", suffix="_1").select("qid", ((pl.col("ce") + pl.col("ce_1")) / 2).alias("ce"))
    v.write_parquet("data/ce_val.parquet"); t.write_parquet("data/ce_test.parquet")
    log("merged", v.height, t.height)


if __name__ == "__main__":
    {"prep": prep, "train": lambda: train(int(sys.argv[2])), "merge": merge}[sys.argv[1]]()
