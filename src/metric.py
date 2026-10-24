"""Macro F0.5 per Source-1 entity, singletons included (empty pred on empty truth = 1)."""


def f05(pred, truth, s1_ids, beta=0.5):
    """pred, truth: dict s1_id -> set of matched ids. s1_ids: ids to evaluate."""
    b2 = beta * beta
    tot = 0.0
    for s in s1_ids:
        p = pred.get(s, set()); t = truth.get(s, set())
        if not p and not t:
            tot += 1.0
            continue
        if not p or not t:
            continue
        tp = len(p & t)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(t)
        tot += (1 + b2) * pr * rc / (b2 * pr + rc)
    return tot / len(s1_ids)
