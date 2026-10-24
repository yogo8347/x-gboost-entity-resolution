"""Candidate generation: IDF-weighted inverted index from S2/S3 query records to S1.

Keys (all prefixed by country, so blocking never crosses countries):
  n:<tok>        name token (core + DBA part)
  k:<skeleton>   consonant skeleton of name token (typos / transliteration)
  a:<num>|<word> street-number x address-word pair (catches renamed / DBA records)
For each query only its R rarest keys (df <= CAP) are probed (prefix-filtering idea),
then the top-K S1 records by summed IDF are kept.
"""
import math
import os
import re
import polars as pl

from normalize import GENERIC_ADDR, LEGAL


def _skel_expr(col):
    first = pl.col(col).str.slice(0, 1)
    rest = pl.col(col).str.slice(1).str.replace_all(r"[aeiouyhw]", "")
    return first + rest


def _phon_expr(col):
    """soundex-like code without first letter (robust to Hindi back-transliteration)."""
    e = pl.col(col).str.replace_all("ph", "f")
    for pat, rep in (("[aeiouyhw]", ""), ("[bfpv]", "1"), ("[cgjkqsxz]", "2"), ("[dt]", "3"),
                     ("l", "4"), ("[mn]", "5"), ("r", "6")):
        e = e.str.replace_all(pat, rep)
    for d in "123456":
        e = e.str.replace_all(d + "+", d)
    return e.str.slice(0, 4)


def make_keys(df):
    """df: normalised records with row id `rid`. Returns long frame (rid:u32, key:u64)."""
    base = df.select(pl.col("rid").cast(pl.UInt32), "country", "core", "dba", "nums", "awords")
    if os.environ.get("BLOCK_TR") == "1":
        # translate garbled / transliterated words with the dictionary learned from TRAINING-fold true pairs
        # (src/translit_dict.py) before building keys: "mharastr" -> "maharashtra", "dijirrl" -> "digital"
        global _TR
        if "_TR" not in globals():
            import json
            _TR = json.load(open("data/translit_dict.json"))
        tr = lambda col, d: (pl.col(col).str.split(" ").list.eval(pl.element().replace(d)).list.join(" ").alias(col))
        base = base.with_columns(tr("core", _TR["core"]), tr("dba", _TR["core"]), tr("awords", _TR["addr"]))
    toks = (base.select("rid", "country",
                        pl.concat_str(["core", "dba"], separator=" ").str.split(" ").alias("t"))
            .explode("t")
            .with_columns(  # leetspeak typos: re1iable, c0nsultancy
                pl.when(pl.col("t").str.contains("[a-z]") & pl.col("t").str.contains("[0-9]"))
                .then(pl.col("t").str.replace_all("0", "o").str.replace_all("1", "l")
                      .str.replace_all("3", "e").str.replace_all("5", "s"))
                .otherwise(pl.col("t")).alias("t"))
            .filter(pl.col("t").str.len_chars() >= 2)
            .filter(~pl.col("t").is_in(list(LEGAL)))
            .unique(["rid", "t"]))
    words = (base.select("rid", "country", pl.col("awords").str.split(" ").alias("w"))
             .explode("w").filter(pl.col("w").str.len_chars() >= 3)
             .filter(~pl.col("w").is_in(list(GENERIC_ADDR))).unique(["rid", "w"]))
    words = words.group_by("rid", maintain_order=True).head(8)
    c = pl.col("country")
    nkeys = toks.select("rid", (c + "|n:" + pl.col("t")).alias("key"))
    kkeys = (toks.filter(pl.col("t").str.len_chars() >= 4)
             .select("rid", (c + "|k:" + _skel_expr("t")).alias("key")))
    tl = toks.group_by("rid", maintain_order=True).agg("country", pl.col("t"))
    tl = tl.with_columns(pl.col("country").list.first())
    # concatenated name prefixes (domain-style names: corinneconnorstrategic.com)
    jkeys = pl.concat([
        tl.filter(pl.col("t").list.len() >= k).select(
            "rid", (c + "|j:" + pl.col("t").list.head(k).list.join("")).alias("key")) for k in (2, 3, 4)]
        + [toks.filter(pl.col("t").str.len_chars() >= 9).select("rid", (c + "|j:" + pl.col("t")).alias("key"))])
    ph = toks.select("rid", "country", _phon_expr("t").alias("p")).filter(pl.col("p").str.len_chars() >= 2)
    pkeys = ph.select("rid", (c + "|p:" + pl.col("p")).alias("key"))
    # ordered pair of phonetic codes of the first two tokens: rare and transliteration-robust
    p2 = (tl.select("rid", "country", pl.col("t").list.head(2).list.eval(
        pl.element().str.replace_all("ph", "f")).alias("t")).filter(pl.col("t").list.len() == 2)
          .explode("t").with_columns(_phon_expr("t").alias("p"))
          .group_by("rid", maintain_order=True).agg(pl.col("country").first(), pl.col("p").str.join("|")))
    p2keys = p2.select("rid", (c + "|q:" + pl.col("p")).alias("key"))
    fkeys = (base.filter(pl.col("core").str.len_chars() >= 3)
             .select("rid", (c + "|f:" + pl.col("core")).alias("key")))
    # name token x address word: rare even when both parts are common
    ckeys = (toks.select("rid", "country", "t").join(words.select("rid", "w"), on="rid")
             .select("rid", (c + "|c:" + pl.col("t") + "|" + pl.col("w")).alias("key")))
    words2 = (base.select("rid", pl.col("awords").str.split(" ").alias("w"))
              .explode("w").filter(pl.col("w").str.len_chars() >= 2)
              .filter(~pl.col("w").is_in(list(GENERIC_ADDR))).unique(["rid", "w"])
              .group_by("rid", maintain_order=True).head(10))
    nums = (base.select("rid", "country", pl.col("nums").str.split(" ").list.head(2).alias("num"))
            .explode("num").filter(pl.col("num").str.len_chars() >= 1))
    a = nums.join(words2, on="rid")
    # house number x first 3 letters of each name token (typos in every address word)
    mkeys = (nums.join(toks.select("rid", pl.col("t").str.slice(0, 3).alias("pfx")), on="rid")
             .select("rid", (c + "|m:" + pl.col("num") + "|" + pl.col("pfx")).alias("key")))
    # unordered pairs of address words (renamed businesses at the same street + city)
    w5 = words.group_by("rid", maintain_order=True).head(5)
    wkeys = (w5.join(w5.select("rid", pl.col("w").alias("w2")), on="rid").filter(pl.col("w") < pl.col("w2"))
             .select("rid", (c + "|w:" + pl.col("w") + "|" + pl.col("w2")).alias("key")))
    akeys = a.select("rid", (c + "|a:" + pl.col("num") + "|" + pl.col("w")).alias("key"))
    # consecutive address word bigrams (Indian addresses often lack numbers)
    bkeys = (base.select("rid", "country", pl.col("awords").str.split(" ").alias("w"))
             .with_columns(pl.col("w").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)))
             .with_columns(pl.col("w").list.slice(1).alias("w2"), pl.col("w").list.head(pl.col("w").list.len() - 1).alias("w"))
             .explode(["w", "w2"]).drop_nulls()
             .select("rid", (c + "|b:" + pl.col("w") + "|" + pl.col("w2")).alias("key")))
    fam = [nkeys, kkeys, fkeys, ckeys, akeys, bkeys, jkeys, pkeys, p2keys, mkeys, wkeys]
    if os.environ.get("BLOCK_DEL") == "1":
        # symmetric-delete keys: every name token (>= 5 chars) with one letter removed. Two tokens within edit
        # distance ~2 share a variant ("binsro" - n = "bisro" = "bistro" - t), which recovers typo'd names that have
        # no address and whose other words are too common to be useful keys.
        long = toks.filter(pl.col("t").str.len_chars() >= 5).select("rid", "country", "t")
        dl = (long.with_columns(pl.int_ranges(0, pl.col("t").str.len_chars()).alias("i")).explode("i")
              .select("rid", "country", (pl.col("t").str.slice(0, pl.col("i")) + pl.col("t").str.slice(pl.col("i") + 1)).alias("v"))
              .unique(["rid", "v"]))
        fam.append(dl.select("rid", (c + "|d:" + pl.col("v")).alias("key")))
    return (pl.concat(fam)
            .select("rid", pl.col("key").hash(seed=17).alias("key")).unique())


def build_index(s1):
    k = make_keys(s1)
    dfreq = k.group_by("key").agg(pl.len().alias("df"))
    return k, dfreq


def generate(s1_keys, dfreq, queries, n1, cap=500, rare=12, topk=8, pre=50, chunk=50_000, log=False):
    """queries: normalised S2/S3 records with `qid`. Returns (qid, rid, bscore, brank).
    Stage 1: probe the query's `rare` rarest keys -> top `pre` S1 records.
    Stage 2: rescore those with IDF over ALL shared keys -> keep top `topk`."""
    keep = dfreq.filter(pl.col("df") <= cap)
    post = s1_keys.join(keep.select("key"), on="key")
    allw = dfreq.select("key", (pl.lit(n1).log() - pl.col("df").log()).cast(pl.Float32).alias("w"))
    out = []
    for st in range(0, queries.height, chunk):
        q = queries.slice(st, chunk).rename({"qid": "rid"})
        qk = make_keys(q).rename({"rid": "qid"}).join(dfreq, on="key")
        probe = (qk.filter(pl.col("df") <= cap).sort(["qid", "df"])
                 .group_by("qid", maintain_order=True).head(rare)
                 .with_columns((pl.lit(n1).log() - pl.col("df").log()).alias("w")))
        cand = (probe.select("qid", "key", "w").join(post, on="key")
                .group_by(["qid", "rid"]).agg(pl.col("w").sum().alias("p")))
        cand = cand.sort(["qid", "p"], descending=[False, True]).group_by("qid", maintain_order=True).head(pre)
        # stage 2: all shared keys (any df) between query and candidate
        qall = qk.select("qid", "key").join(allw, on="key")
        shared = (cand.select("qid", "rid").join(qall, on="qid")
                  .join(s1_keys, on=["rid", "key"], how="semi")
                  .group_by(["qid", "rid"]).agg(pl.col("w").sum().alias("bscore")))
        pairs = (shared.sort(["qid", "bscore"], descending=[False, True])
                 .group_by("qid", maintain_order=True).head(topk)
                 .with_columns(pl.int_range(pl.len()).over("qid").cast(pl.UInt8).alias("brank")))
        out.append(pairs)
        if log and (st // chunk) % 20 == 0:
            print(f"  block chunk {st}/{queries.height} pairs so far {sum(o.height for o in out)}", flush=True)
    return pl.concat(out)
