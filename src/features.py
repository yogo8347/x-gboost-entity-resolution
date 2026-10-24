"""Pair features for (query S2/S3 record, candidate S1 record).

Deliberately country-agnostic (no country one-hot) so the model transfers to unseen
countries such as France: only similarity scores, number agreement and competition
features relative to the other candidates of the same query / S1 record.
"""
import os
import re
import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz

VOW = re.compile(r"[aeiouyhw]")


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _skel(s):
    return " ".join(t[0] + VOW.sub("", t[1:]) for t in s.split() if t)


def _skel_vec(a):
    s = pl.Series(a).str.replace_all(r"\b([a-z])[aeiouyhw]+", "$1").str.replace_all(r"\B[aeiouyhw]", "")
    return s.to_list()


def _set_feats(a, b):
    """token jaccard / containment on space separated strings (vectorised)."""
    d = pl.DataFrame({"a": a, "b": b}).select(
        pl.col("a").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("x"),
        pl.col("b").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("y"))
    d = d.select(pl.col("x").list.set_intersection("y").list.len().cast(pl.Float32).alias("i"),
                 pl.col("x").list.len().cast(pl.Float32).alias("nx"),
                 pl.col("y").list.len().cast(pl.Float32).alias("ny"))
    ok = (d["nx"] > 0) & (d["ny"] > 0)
    i, nx, ny = d["i"].to_numpy(), d["nx"].to_numpy(), d["ny"].to_numpy()
    okn = ok.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(okn, i / (nx + ny - i), 0).astype(np.float32)
        cq = np.where(okn, i / nx, 0).astype(np.float32)
        cs = np.where(okn, i / ny, 0).astype(np.float32)
    return jac, cq, cs


def _num_feats(qn, sn):
    n = len(qn)
    first_eq = np.zeros(n, np.float32); any_sh = np.zeros(n, np.float32)
    s_in_q = np.zeros(n, np.float32); conflict = np.zeros(n, np.float32)
    first_close = np.zeros(n, np.float32)
    for i in range(n):
        a = qn[i].split(); b = sn[i].split()
        if not a or not b:
            first_eq[i] = any_sh[i] = s_in_q[i] = conflict[i] = first_close[i] = -1
            continue
        first_eq[i] = a[0] == b[0]
        # truncated / typo'd house numbers: 387 vs 38, 235 vs 35
        first_close[i] = a[0] in b[0] or b[0] in a[0] or Levenshtein.distance(a[0], b[0]) <= 1
        sa, sb = set(a), set(b)
        any_sh[i] = len(sa & sb) > 0
        s_in_q[i] = len(sa & sb) / len(sb)
        conflict[i] = (a[0] not in sb) and (b[0] not in sa)
    return first_eq, first_close, any_sh, s_in_q, conflict


LEGALISH = (r"\b(?:pr[a-z]{0,3}[bv]h?[ae]?t[a-z]?|priv[a-z]*|pvt|lim[a-z]{0,4}|ltd|llp|llc|inc|corp|co|"
            r"m|s|ms|messrs|sri|shri|smt|the|and|of|www|com|in|org|net|pllc|pc|pa|lp|group|grp)\b")


def _clean_vec(a):
    """remove legal-ish tokens incl. transliterated spellings (praibhet, limitet) and honorifics."""
    s = pl.Series(a).str.replace_all(LEGALISH, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    return s.to_list()


def _phon_vec(a):
    """per-token phonetic code (soundex-like, no first letter) joined -> robust to transliteration."""
    e = pl.Series(a).str.replace_all("ph", "f")
    for pat, rep in (("[bfpv]", "1"), ("[cgjkqsxz]", "2"), ("[dt]", "3"), ("l", "4"), ("[mn]", "5"),
                     ("r", "6"), ("[aeiouyhw]", "")):
        e = e.str.replace_all(pat, rep)
    for d in "123456":
        e = e.str.replace_all(d + "+", d)
    return e.to_list()


def _num_feats2(qn, sn):
    n = len(qn)
    all_eq = np.full(n, -1, np.float32); q_extra = np.full(n, -1, np.float32)
    s_extra = np.full(n, -1, np.float32); absdiff = np.full(n, -1, np.float32)
    for i in range(n):
        a = qn[i].split(); b = sn[i].split()
        if not a or not b:
            continue
        sa, sb = set(a), set(b)
        all_eq[i] = sa == sb
        q_extra[i] = len(sa - sb); s_extra[i] = len(sb - sa)
        try:
            absdiff[i] = np.log1p(abs(int(a[0][:9]) - int(b[0][:9])))
        except ValueError:
            pass
    return all_eq, q_extra, s_extra, absdiff


def _derive(df, dba):
    core = df["core"].fill_null("").to_list()
    clean = _clean_vec(core)
    out = {"skel": _skel_vec(core), "clean": clean, "phon": _phon_vec(clean)}
    if dba:
        out["dclean"] = _clean_vec(df["dba"].fill_null("").to_list())
    return df.select(df.columns[0]).with_columns([pl.Series(k, v) for k, v in out.items()])


_DUP = {}


def compute(pairs, q, s1):
    """pairs: (qid, rid, bscore, brank). q has qid, s1 has rid. Returns feature frame."""
    cols = ["name", "core", "dba", "addr", "nums", "awords"]
    # regex-heavy transforms computed once per RECORD (not per pair), then joined onto the pairs
    qr = _derive(q.join(pairs.select("qid").unique(), on="qid", how="semi").select("qid", "core", "dba"), dba=True)
    sr = _derive(s1.join(pairs.select("rid").unique(), on="rid", how="semi").select("rid", "core", "dba"), dba=False)
    d = (pairs.join(q.select("qid", "entity_id", *cols), on="qid")
         .join(s1.select("rid", *[pl.col(c).alias("s_" + c) for c in cols]), on="rid")
         .join(qr, on="qid").join(sr.rename({c: "s_" + c for c in sr.columns if c != "rid"}), on="rid"))
    g = lambda c: d[c].fill_null("").to_list()
    qc, sc, qd = g("core"), g("s_core"), g("dba")
    qf, sf = g("name"), g("s_name")
    qa, sa = g("addr"), g("s_addr")
    qw, sw = g("awords"), g("s_awords")
    f = {
        "bscore": d["bscore"].to_numpy().astype(np.float32),
        "brank": d["brank"].to_numpy().astype(np.float32),
        "is_s3": d["entity_id"].str.starts_with("S3").to_numpy().astype(np.float32),
        "n_ratio": _cp(fuzz.ratio, qc, sc),
        "n_tset": _cp(fuzz.token_set_ratio, qc, sc),
        "n_tsort": _cp(fuzz.token_sort_ratio, qc, sc),
        "n_partial": _cp(fuzz.partial_ratio, qc, sc),
        "n_jw": _cp(JaroWinkler.normalized_similarity, qc, sc),
        "n_full_ratio": _cp(fuzz.ratio, qf, sf),
        "n_dba_ratio": _cp(fuzz.ratio, qd, sc),
        "n_dba_tset": _cp(fuzz.token_set_ratio, qd, sc),
        "n_skel_ratio": _cp(fuzz.token_sort_ratio, g("skel"), g("s_skel")),
        "n_len_q": np.array([len(x) for x in qc], np.float32),
        "n_len_s": np.array([len(x) for x in sc], np.float32),
        "a_q_empty": np.array([not x for x in qa], np.float32),
        "a_tset": _cp(fuzz.token_set_ratio, qa, sa),
        "a_tsort": _cp(fuzz.token_sort_ratio, qa, sa),
        "a_partial": _cp(fuzz.partial_ratio, qa, sa),
        "w_tset": _cp(fuzz.token_set_ratio, qw, sw),
    }
    qcl, scl = g("clean"), g("s_clean")
    qdl = g("dclean")
    qph, sph = g("phon"), g("s_phon")
    f["n_clean_ratio"] = _cp(fuzz.ratio, qcl, scl)
    f["n_clean_tset"] = _cp(fuzz.token_set_ratio, qcl, scl)
    f["n_clean_dba_ratio"] = _cp(fuzz.ratio, qdl, scl)
    f["n_clean_eq"] = np.array([a == b and a != "" for a, b in zip(qcl, scl)], np.float32)
    f["n_clean_lev"] = _cp(Levenshtein.distance, qcl, scl)
    f["n_nospace_ratio"] = _cp(fuzz.ratio, [x.replace(" ", "") for x in qcl], [x.replace(" ", "") for x in scl])
    f["n_phon_ratio"] = _cp(fuzz.ratio, qph, sph)
    f["n_phon_tsort"] = _cp(fuzz.token_sort_ratio, qph, sph)
    f["n_lev"] = _cp(Levenshtein.distance, qc, sc)
    f["a_ratio"] = _cp(fuzz.ratio, qa, sa)
    f["a_s_empty"] = np.array([not x for x in sa], np.float32)
    (f["num_all_eq"], f["num_q_extra"], f["num_s_extra"], f["num_first_absdiff"]) = _num_feats2(g("nums"), g("s_nums"))
    f["n_jac"], f["n_cq"], f["n_cs"] = _set_feats(qc, sc)
    f["nd_jac"], _, f["nd_cs"] = _set_feats(qd, sc)
    f["w_jac"], f["w_cq"], f["w_cs"] = _set_feats(qw, sw)
    (f["num_first_eq"], f["num_first_close"], f["num_any"], f["num_s_in_q"],
     f["num_conflict"]) = _num_feats(g("nums"), g("s_nums"))
    if os.environ.get("STAGE1_V10") == "1":
        # v10: the decoy / word / clean / translated markers (previously only in stage 2) become stage-1 features,
        # so the pair model itself ranks and scores candidates with them
        from decoy_markers import markers
        from clean_markers import clean_markers
        from translit_markers import DN, DA, tr, clean_name, clean_addr
        f.update(markers(g("name"), g("s_name"), g("nums"), g("s_nums"), qc, sc))
        f.update(clean_markers(g("name"), g("s_name"), qa, sa))
        qn2 = [clean_name(tr(x, DN)) for x in qc]; sn2 = [clean_name(tr(x, DN)) for x in sc]
        qa2 = [clean_addr(tr(x, DA)) for x in qa]; sa2 = [clean_addr(tr(x, DA)) for x in sa]
        f["tr_n_ratio"] = _cp(fuzz.ratio, qn2, sn2); f["tr_n_tset"] = _cp(fuzz.token_set_ratio, qn2, sn2)
        f["tr_a_tset"] = _cp(fuzz.token_set_ratio, qa2, sa2); f["tr_a_ratio"] = _cp(fuzz.token_sort_ratio, qa2, sa2)
    out = pl.DataFrame({"qid": d["qid"], "rid": d["rid"], **f})
    # name rarity: how many S1 records carry exactly this S1's / this query's core name
    key = (id(s1), s1.height)
    if key not in _DUP:
        _DUP.clear()
        _DUP[key] = s1.group_by("core").agg(pl.len().cast(pl.Float32).alias("dup"))
    dup = _DUP[key]
    out = out.with_columns(pl.Series("_sc", sc), pl.Series("_qc", qc))
    out = (out.join(dup.rename({"core": "_sc", "dup": "s_name_dup"}), on="_sc", how="left")
           .join(dup.rename({"core": "_qc", "dup": "q_name_dup"}), on="_qc", how="left")
           .drop("_sc", "_qc").with_columns(pl.col("s_name_dup").fill_null(0), pl.col("q_name_dup").fill_null(0)))
    # competition features: how this candidate compares with the query's other candidates
    comb = (pl.col("n_tset") + pl.col("a_tset")).alias("comb")
    out = out.with_columns(comb, pl.len().over("qid").alias("q_ncand").cast(pl.Float32))
    out = out.with_columns((pl.col("n_clean_ratio") + pl.col("n_phon_ratio") + pl.col("a_tset")
                            + 50 * pl.col("num_first_eq").clip(0, 1) - 50 * pl.col("num_conflict").clip(0, 1)).alias("comb2"))
    for c in ("comb", "comb2", "n_tset", "a_tset", "bscore", "n_clean_ratio", "n_phon_ratio"):
        out = out.with_columns(
            (pl.col(c) - pl.col(c).max().over("qid")).alias(f"{c}_gap"),
            (pl.col(c) - pl.col(c).filter(pl.col(c) < pl.col(c).max()).max().over("qid").fill_null(0))
            .alias(f"{c}_gap2"))
    out = out.with_columns(pl.col("comb").rank("ordinal", descending=True).over("qid")
                           .cast(pl.Float32).alias("comb_rank"),
                           pl.col("comb2").rank("ordinal", descending=True).over("qid")
                           .cast(pl.Float32).alias("comb2_rank"))
    extra = [c for c in pairs.columns if c.startswith("x_")]   # v11: trigram-leg evidence carried as features
    if extra:
        out = out.join(pairs.select("qid", "rid", *extra), on=["qid", "rid"], how="left")
    if "s_ncand" in pairs.columns:  # computed on the FULL candidate table by the pipeline
        out = out.join(pairs.select("qid", "rid", "s_ncand"), on=["qid", "rid"])
    return out


def feature_names(df):
    return [c for c in df.columns if c not in ("qid", "rid", "label")]
