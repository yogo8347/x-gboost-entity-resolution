"""Clean-comparison markers (v7), from the full vocabulary audit (data/vocab_audit.txt).

Both the record and its best S1 are cleaned the same way, then compared again:
  names:     legal forms (incl. garbled transliterations such as praivrr / limirrd), honorifics (mr, dr, sri, shri, smt),
             injected US area filler words (westgate, riverside, midtown, ...), alias markers; leetspeak repaired
             (hea1th -> health, lnc -> inc), tek -> technology
  addresses: Indian state names / codes / transliterations (mh, mharastr, dilli, krnatk, tmilnatu, pscimbng, ...)
             dropped like US state names are unified; US noise tokens and their typos dropped (city / cdp / township
             variants, po box, pmb); French 'no' / 'ndeg'; ordinal words turned into digits (fourth -> 4)
Also extends the decoy-word list with the French decoy words found in the audit (words that are EXTRA in borderline
France matches but never in 818k confident ones): snc, international, participations, distribution, holding,
groupement, etablissements, ets, ei.

  python src/clean_markers.py <table_in> <table_out> <prep_dir> <split>
"""
import re
import sys
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "src")
from normalize import LEGAL
from word_markers import DECOY_EXTRA, ALIAS

FR_DECOY = {"snc", "international", "participations", "distribution", "holding", "groupement", "etablissements", "ets", "ei"}
HONORIFIC = {"mr", "dr", "mrs", "ms", "sri", "shri", "smt", "m", "s"}
US_FILLER = {"westgate", "southside", "lakeside", "riverside", "midtown", "northside", "eastgate", "greater", "downtown",
             "uptown", "central", "district", "authority", "sys", "the"}
TRANSLIT_LEGAL = re.compile(r"^(pr[a-z]{0,4}(t|rr|d)|lim[a-z]{0,5}|piraivet|elelpi|pra|li)$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s"})
IN_STATE = {"mh", "maharashtra", "mharastr", "maharastra", "dl", "dilli", "ka", "karnataka", "krnatk", "up", "uttar", "uttr",
            "pradesh", "prdes", "tn", "tamil", "nadu", "tmilnatu", "wb", "bengal", "pscimbng", "gj", "gujarat", "gujrat",
            "tg", "telangana", "telmgan", "hr", "haryana", "hriyana", "rj", "rajasthan", "rajsthan", "kl", "kerala",
            "keralam", "kerlm", "mp", "madhya", "mdhy", "br", "bihar", "od", "odisha", "ap", "andhra", "amdhrprdes",
            "pb", "punjab", "pmjab", "region", "divreportingcircle", "isa", "hq"}
ADDR_NOISE = {"po", "box", "pmb", "of", "door", "dor", "doro", "hn", "h", "no", "ndeg", "n"}
ORD = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
       "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
       "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19", "twentieth": "20"}


def _is_place_noise(t):
    if 2 <= len(t) <= 4 and set(t) <= set("cdp"):                     # cdp, dcp, ccdp, cddp, cpd, cp, dp (never a single block letter)
        return True
    if 3 <= len(t) <= 6 and Levenshtein.distance(t, "city") <= 2 and t not in {"cit", "ci"}:   # ity, ciy, citty, icty ...
        return len(set(t) - set("city")) == 0
    if 6 <= len(t) <= 10 and Levenshtein.distance(t, "township") <= 2:  # ownship, towship, townshp, ttownship ...
        return True
    return False


def clean_name(s):
    out = []
    for t in (s or "").split():
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            t = t.translate(LEET)
        t = {"lnc": "inc", "tek": "technology"}.get(t, t)
        if t in LEGAL or t in HONORIFIC or t in US_FILLER or t in ALIAS or TRANSLIT_LEGAL.match(t):
            continue
        out.append(t)
    return " ".join(out)


def clean_addr(s):
    out = []
    for t in (s or "").split():
        t = ORD.get(t, t)
        if t in IN_STATE or t in ADDR_NOISE or _is_place_noise(t):
            continue
        out.append(t)
    return " ".join(out)


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def clean_markers(qn, sn, qa, sa):
    cqn = [clean_name(x) for x in qn]; csn = [clean_name(x) for x in sn]
    cqa = [clean_addr(x) for x in qa]; csa = [clean_addr(x) for x in sa]
    f = {"cn_ratio": _cp(fuzz.ratio, cqn, csn), "cn_tset": _cp(fuzz.token_set_ratio, cqn, csn),
         "cn_eq": np.array([a == b and a != "" for a, b in zip(cqn, csn)], np.float32),
         "ca_tset": _cp(fuzz.token_set_ratio, cqa, csa), "ca_ratio": _cp(fuzz.token_sort_ratio, cqa, csa),
         "ca_eq": np.array([set(a.split()) == set(b.split()) and a != "" for a, b in zip(cqa, csa)], np.float32)}
    dec = np.zeros(len(qn), np.float32)
    for i, (a, b) in enumerate(zip(qn, sn)):
        ex = set((a or "").split()) - set((b or "").split())
        dec[i] = len(ex & (DECOY_EXTRA | FR_DECOY))
    f["wm_decoy_extra2"] = dec
    return f


if __name__ == "__main__":
    tin, tout, prep, split = sys.argv[1:5]
    t = pl.read_parquet(tin)
    q = (pl.concat([pl.read_parquet(f"{prep}/{split}_s{k}.parquet", columns=["name", "addr"]) for k in (2, 3)])
         .with_row_index("qid").with_columns(pl.col("qid").cast(pl.UInt32)))
    s1 = pl.read_parquet(f"{prep}/{split}_s1.parquet", columns=["name", "addr"]).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
    p = t.select("qid", "rid").join(q, on="qid", how="left").join(s1.rename({"name": "s_name", "addr": "s_addr"}), on="rid", how="left")
    g = lambda c: p[c].fill_null("").to_list()
    m = clean_markers(g("name"), g("s_name"), g("addr"), g("s_addr"))
    t = t.with_columns([pl.Series(k, v) for k, v in m.items()])
    t.write_parquet(tout)
    print(tout, t.shape, {k: round(float(v.mean()), 3) for k, v in m.items()})
