"""Decoy markers: pairwise signals that separate hard-negative decoys from genuinely noisy copies.

Measured on the held-out fold (accepted-range pairs, p1 > 0.3):
  * legal-form conflict (e.g. Inc vs LLC):          true 0.3%  vs decoys 13.9%   (46x)
  * record carries a legal form its S1 lacks:        true 6.6%  vs decoys 31.0%
  * first house number: one digit substituted:       true 1.2%  vs decoys 9.8%    (8.3x)
  * first house number: completely different:        true 1.2%  vs decoys 10.8%   (8.9x)
Name cleaning removes legal forms from the core name, so the pair model could not see the first two at all.
All markers compare one record with one S1 record: they do not depend on how many distractors a dataset has.
"""
import re
import numpy as np
from rapidfuzz.distance import Levenshtein

LEG = {"inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "llc": "llc", "lc": "llc", "lp": "lp",
       "llp": "llp", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "co": "co", "company": "co",
       "pllc": "pllc", "pc": "pc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
       "sci": "sci", "snc": "snc", "sa": "sa", "ei": "ei"}
NUM_REL = ("equal", "reordered", "missing", "truncated", "one_digit", "different")
TOK = re.compile(r"[a-z]+")


def _forms(name):
    return {LEG[t] for t in TOK.findall(name or "") if t in LEG}


def _num_rel(qn, sn):
    A, B = (qn or "").split(), (sn or "").split()
    if not A or not B:
        return "missing"
    x, y = A[0], B[0]
    if x == y:
        return "equal"
    if x in B or y in A:
        return "reordered"
    if len(x) == len(y) and sum(a != b for a, b in zip(x, y)) == 1:
        return "one_digit"
    if x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x):
        return "truncated"
    return "different"


def _word_changes(qc, sc):
    A, B = (qc or "").split(), (sc or "").split()
    ua = [x for x in A if x not in B]; ub = [x for x in B if x not in A]
    typo = sub = 0
    for x in ua:
        d = min((Levenshtein.distance(x, y) for y in ub), default=99)
        if d <= 2 and len(x) >= 3:
            typo += 1
        else:
            sub += 1
    return typo, sub, len(ub)


def markers(q_name, s_name, q_nums, s_nums, q_core, s_core):
    """lists of normalised strings (one entry per pair) -> dict of float32 feature arrays."""
    n = len(q_name)
    out = {k: np.zeros(n, np.float32) for k in
           ("dm_legal_conflict", "dm_legal_q_only", "dm_legal_s_only", "dm_word_typo", "dm_word_sub", "dm_s_word_missing")}
    for r in NUM_REL:
        out["dm_num_" + r] = np.zeros(n, np.float32)
    for i in range(n):
        fq, fs = _forms(q_name[i]), _forms(s_name[i])
        out["dm_legal_conflict"][i] = bool(fq) and bool(fs) and not (fq & fs)
        out["dm_legal_q_only"][i] = bool(fq - fs)
        out["dm_legal_s_only"][i] = bool(fs - fq)
        out["dm_num_" + _num_rel(q_nums[i], s_nums[i])][i] = 1.0
        t, s, m = _word_changes(q_core[i], s_core[i])
        out["dm_word_typo"][i], out["dm_word_sub"][i], out["dm_s_word_missing"][i] = t, s, m
    return out
