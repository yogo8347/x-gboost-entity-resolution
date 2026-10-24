# X_Gboost - cross-encoder v4: bge-reranker-v2-m3, FULL epoch on BOTH T4 GPUs (DataParallel). Uses dataset xgboost-ce2.
# Same design as v2 (v12): ONE model trained on stage-1 TRAINING-fold pairs (never a held-out S1),
# then scores ALL validation pairs and ALL test pairs. Outputs /kaggle/working/ce4_val.parquet, ce4_test.parquet
# Needs the dataset with train_pairs.parquet (new) + val_pairs.parquet + test_pairs.parquet (same as last run).
import glob, os, time, numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
MODEL = "BAAI/bge-reranker-v2-m3"          # 568M, Apache-2.0, multilingual
MAXLEN, BS, LR, EPOCHS = 64, 64, 2e-5, 1
TIME_BUDGET = 1.6 * 3600                   # stop training after 1.6 h and go on to scoring
dev = "cuda"; print(torch.cuda.get_device_name(0), torch.cuda.device_count(), "GPU(s)")
f = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr = pl.read_parquet(f("train_pairs.parquet")).sample(fraction=1.0, shuffle=True, seed=0)
val = pl.read_parquet(f("val_pairs.parquet")); test = pl.read_parquet(f("test_pairs.parquet"))
print("train", tr.height, "val", val.height, "test", test.height, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)

def batches(q, s, bs):
    for i in range(0, len(q), bs):
        yield i, tok(q[i:i+bs], s[i:i+bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

def predict(m, q, s):
    m.eval(); out = np.zeros(len(q), np.float32); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i, b in batches(q, s, 1024):
            b = {k: v.to(dev) for k, v in b.items()}
            out[i:i+len(b["input_ids"])] = torch.sigmoid(m(**b)["logits"].float().squeeze(-1)).cpu().numpy()
            if (i // 1024) % 200 == 0: print(f"  predict {i}/{len(q)} {i/(time.time()-t0+1e-9):.0f}/s", flush=True)
    return out

base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, ignore_mismatched_sizes=True).to(dev)
m = torch.nn.DataParallel(base) if torch.cuda.device_count() > 1 else base   # both T4s
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
q, s, y = tr["qt"].to_list(), tr["st"].to_list(), tr["label"].to_numpy().astype(np.float32)
steps = EPOCHS * ((len(y) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
scaler = torch.cuda.amp.GradScaler(); lossf = torch.nn.BCEWithLogitsLoss(); m.train(); t0 = time.time(); st_ = 0
for ep in range(EPOCHS):
    for i, b in batches(q, s, BS):
        b = {k: v.to(dev) for k, v in b.items()}; yy = torch.tensor(y[i:i+BS], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(m(**b)["logits"].float().squeeze(-1), yy)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); st_ += 1
        if st_ % 500 == 0: print(f"step {st_}/{steps} loss {loss.item():.4f} {(st_*BS)/(time.time()-t0):.0f} pairs/s", flush=True)
        if time.time() - t0 > TIME_BUDGET: print("time budget reached at step", st_, flush=True); break
vp = predict(m, val["qt"].to_list(), val["st"].to_list())
print(f"VALIDATION AUC (never seen, held-out S1): {roc_auc_score(val['label'].to_numpy(), vp):.4f}", flush=True)
val.select("qid").with_columns(pl.Series("ce4", vp)).write_parquet("/kaggle/working/ce4_val.parquet")
tp = predict(m, test["qt"].to_list(), test["st"].to_list())
test.select("qid").with_columns(pl.Series("ce4", tp)).write_parquet("/kaggle/working/ce4_test.parquet")
print("DONE")
