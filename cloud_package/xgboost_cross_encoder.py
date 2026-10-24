# X_Gboost — cross-encoder on uncertain best pairs (2-fold, leak-free). Output: /kaggle/working/ce_val.parquet, ce_test.parquet
import glob, os, time, numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "BAAI/bge-reranker-v2-m3"          # 568M, Apache-2.0, multilingual (French / transliterated Hindi)
MAXLEN, BS, LR, EPOCHS = 64, 32, 2e-5, 1
dev = "cuda"; print(torch.cuda.get_device_name(0))
root = os.path.dirname(glob.glob("/kaggle/input/**/val_pairs.parquet", recursive=True)[0])
val = pl.read_parquet(f"{root}/val_pairs.parquet"); test = pl.read_parquet(f"{root}/test_pairs.parquet")
print("val", val.height, "test", test.height)
tok = AutoTokenizer.from_pretrained(MODEL)

def batches(q, s, bs):
    for i in range(0, len(q), bs):
        yield i, tok(q[i:i+bs], s[i:i+bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

def predict(m, q, s):
    m.eval(); out = np.zeros(len(q), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i, b in batches(q, s, 256):
            b = {k: v.to(dev) for k, v in b.items()}
            out[i:i+len(b["input_ids"])] = torch.sigmoid(m(**b).logits.float().squeeze(-1)).cpu().numpy()
    return out

val_pred = np.zeros(val.height, np.float32); test_pred = np.zeros(test.height, np.float32)
for fold in (0, 1):
    fit = val.filter(pl.col("fold") != fold).sample(fraction=1.0, shuffle=True, seed=fold)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, ignore_mismatched_sizes=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    q, s, y = fit["qt"].to_list(), fit["st"].to_list(), fit["label"].to_numpy().astype(np.float32)
    steps = EPOCHS * ((len(y) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06)
    scaler = torch.cuda.amp.GradScaler(); lossf = torch.nn.BCEWithLogitsLoss(); m.train(); t0 = time.time(); st_ = 0
    for ep in range(EPOCHS):
        for i, b in batches(q, s, BS):
            b = {k: v.to(dev) for k, v in b.items()}; yy = torch.tensor(y[i:i+BS], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(m(**b).logits.float().squeeze(-1), yy)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); st_ += 1
            if st_ % 500 == 0: print(f"fold {fold} step {st_}/{steps} loss {loss.item():.4f} {(st_*BS)/(time.time()-t0):.0f} pairs/s", flush=True)
    ho = (val["fold"] == fold).to_numpy()
    val_pred[ho] = predict(m, val.filter(pl.col("fold") == fold)["qt"].to_list(), val.filter(pl.col("fold") == fold)["st"].to_list())
    from sklearn.metrics import roc_auc_score
    print(f"fold {fold} held-out AUC {roc_auc_score(val.filter(pl.col('fold') == fold)['label'].to_numpy(), val_pred[ho]):.4f}", flush=True)
    test_pred += predict(m, test["qt"].to_list(), test["st"].to_list()) / 2
    del m; torch.cuda.empty_cache()
val.select("qid").with_columns(pl.Series("ce", val_pred)).write_parquet("/kaggle/working/ce_val.parquet")
test.select("qid").with_columns(pl.Series("ce", test_pred)).write_parquet("/kaggle/working/ce_test.parquet")
print("DONE")
