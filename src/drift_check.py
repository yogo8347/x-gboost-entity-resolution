"""Drift gate: PSI of every model input between the validation table and the test table.
A feature whose distribution shifts between validation and test (PSI > limit) makes validation scores
untrustworthy on test. Exit code 1 if any input exceeds the limit.

  python src/drift_check.py <final.json> <val_table> <test_table> [limit=0.10]
"""
import json, sys
import numpy as np
import polars as pl

cfg, va_p, te_p = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
limit = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
feats = cfg["feats"]
va = pl.read_parquet(va_p, columns=feats); te = pl.read_parquet(te_p, columns=feats)


def psi(a, b, bins=10):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:  # (near-)constant feature: compare the share of the dominant value instead
        v = np.median(a); pa, pb = np.mean(a == v), np.mean(b == v)
        return float(abs(pa - pb))
    pa = np.histogram(a, qs)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, qs[0], qs[-1]), qs)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


rows = sorted(((f, psi(va[f].cast(pl.Float64).to_numpy(), te[f].cast(pl.Float64).to_numpy())) for f in feats),
              key=lambda x: -x[1])
for f, v in rows[:12]:
    print(f"  {f:28s} PSI={v:.3f}{'   <-- FAIL' if v > limit else ''}")
bad = [f for f, v in rows if v > limit]
print(f"DRIFT GATE: {'PASS' if not bad else 'FAIL'}  ({len(feats)} inputs, max PSI {rows[0][1]:.3f}, limit {limit})")
sys.exit(1 if bad else 0)
