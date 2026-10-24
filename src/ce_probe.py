"""Quick value test for the cross-encoder before investing GPU hours.

Fine-tunes MiniLM on a 64k sample of held-out-fold half A (uncertain band), then compares on 20k pairs of half B:
AUC of the transformer vs AUC of the current stage-2 decision (v8 out-of-fold probabilities) and vs a blend.
Leak-free: training rows (half A) and evaluation rows (half B) belong to different S1 entities.

  python src/ce_probe.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cross_encoder as ce

ce.MAXLEN = 64
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pl.read_parquet("data/ce/train_pairs.parquet").filter((pl.col("p1") >= 0.02) & (pl.col("p1") <= 0.995))
A = tr.filter(pl.col("fold") == 0).sample(64_000, seed=1)
B = tr.filter(pl.col("fold") == 1).sample(20_000, seed=2)
tok = AutoTokenizer.from_pretrained(ce.MODEL)
model = AutoModelForSequenceClassification.from_pretrained(ce.MODEL, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
qt, st, y = A["qt"].to_list(), A["st"].to_list(), A["label"].to_numpy().astype(np.float32)
steps = len(y) // 64
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-5, total_steps=steps + 1, pct_start=0.06)
scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda"); lossf = torch.nn.BCEWithLogitsLoss()
t0 = time.time(); model.train()
for step, (i, b) in enumerate(ce._batches(qt, st, 64, tok)):
    b = {k: v.to(dev) for k, v in b.items()}
    yy = torch.tensor(y[i:i + 64], device=dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        loss = lossf(model(**b).logits.float().squeeze(-1), yy)
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if step % 200 == 0:
        print(time.strftime("%H:%M:%S"), f"step {step}/{steps} loss {loss.item():.4f} ({(step + 1) * 64 / (time.time() - t0):.0f} pairs/s)", flush=True)
model.save_pretrained("data/ce/probe_model"); tok.save_pretrained("data/ce/probe_model")
t1 = time.time()
p = ce.predict(model, tok, B["qt"].to_list(), B["st"].to_list(), dev)
print(f"inference speed: {len(p) / (time.time() - t1):.0f} pairs/s", flush=True)
oof = pl.read_parquet("data/v5_oof.parquet").select("qid", pl.col("prob").alias("s2"))
Bj = B.select("qid", "label", "p1").with_columns(pl.Series("ce", p)).join(oof, on="qid", how="left")
lab = Bj["label"].to_numpy()
auc = lambda x: roc_auc_score(lab, x)
s2 = Bj["s2"].fill_null(0.5).to_numpy()
print(f"AUC on 20k held-out uncertain pairs:  stage-1 p1 {auc(Bj['p1'].to_numpy()):.4f} | current stage 2 {auc(s2):.4f} | "
      f"transformer {auc(Bj['ce'].to_numpy()):.4f} | blend (s2+ce)/2 {auc((s2 + Bj['ce'].to_numpy()) / 2):.4f}")
