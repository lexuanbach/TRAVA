#!/usr/bin/env python3
"""R4: selection-refit inference for E5, symmetric decision table, and the
incremental value of L2 masking within the date-conditioned policy class.

Cached outputs only, CPU (numpy), no network. Nothing in e5_decision.py is
changed: the candidate policies (E.POLICIES), candidate boundaries (E.BOUNDS),
selection rule (quiz-weighted dev mean F1), tie-breaking (exact rational ties
to the earliest candidate in E.POLICIES order), fold construction for the
original data (E.folds_for) and the endpoint (cross-fitted per-question F1,
audit minus naive) are imported and reproduced. The script first checks that
its vectorised re-implementation reproduces every one of the stored 100 fits
per generator and the stored point estimates and CIs.

1. Selection-refit (outer) bootstrap. The stored E5 CIs resample weekly quizzes
   over per-question scores that were already cross-fitted and averaged, i.e.
   they are conditional on the fitted selections. Here each of B draws
   (default 1000, numpy default_rng(20260918)) resamples the 199 weekly quizzes
   with replacement and reruns the WHOLE procedure on the resampled data set:
   R=5 repeats x 5 folds of quiz-grouped cross-fitting, where the fold unit is
   the distinct original quiz, so every copy of a resampled quiz lands in the
   same fold (no copy of a test quiz is ever in its own dev set); within each
   repeat the distinct quizzes are put in a random order and cut into 5
   contiguous near-equal groups, the same cut rule as E.folds_for. Selection in
   a resampled data set maximises the quiz-weighted dev mean over quiz copies
   (multiplicity-weighted), with exact ties broken as in E5 (float ranking,
   structural-tie detection, and an exact Fraction re-evaluation of any
   remaining near-tie). The recorded statistic is the cross-fitted
   quiz-weighted and question-pooled audit-naive difference in the resampled
   data set. Reported: percentile 95% CI, two-sided plus-one bootstrap p
   (analysis_common.summarize), Holm across the nine generators on the
   quiz-weighted p, and the share of draws in which audit and naive take
   identical held-out actions in every fit. The point estimate is from the
   original data with the same 5-repeat procedure (repeats 0-4 of E.folds_for,
   a subset of the stored 20); the stored 20-repeat value is shown alongside.

2. Symmetric table. Verdict categories: "unchanged actions" (audit and naive
   take identical held-out retrieval actions in every one of the stored 100
   fits), otherwise "significant gain"/"significant loss" (Holm p < 0.05, sign
   of the original-data estimate), otherwise "unresolved". Applied once with
   the stored conditional p and once with the refit p.

3. Incremental value of L2 masking. Two contrasts, both for the audit-informed
   procedure (and the naive one for reference):
   (a) own-arm scoring, mirroring the stored leakage variant: procedure run on
       masked RAG outputs minus the same procedure run on unmasked `hybrid`
       outputs (CB and DATE_b closed-book branches unchanged). Unmasked outputs
       may use post-question evidence, so (a) is not a deployable contrast.
   (b) decision value: the policy selected on masked outputs minus the policy
       selected on unmasked outputs, BOTH scored on the masked (admissible)
       held-out outputs. Zero whenever masking does not change the held-out
       retrieval actions. Holm within each secondary contrast family.
   Both get the stored-style conditional CI (20 repeats, QuizBoot B=5000, seed
   20260917) and the refit CI.

Writes results/analysis/r4_e5_outer.json and results/analysis/r4_e5_table.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_common as C  # noqa: E402
import e5_decision as E  # noqa: E402
from routing_baselines import holm  # noqa: E402

SEED_OUTER = 20260918
B_OUTER = 1000
R_OUTER = 5
R_KEYS = 20                      # fold keys drawn for 20 repeats; the first R used
F = E.FOLDS
P0 = len(E.POLICIES)             # 10 base policies
ARMS = ("masked", "unmasked")
RAG_COND = {"masked": "hybrid_masked", "unmasked": "hybrid"}
NEAR = 1e-6                      # float near-tie window, resolved exactly
ZERO = 1e-9
ALPHA = E.ALPHA
STORED = C.OUT / "e5_decision.json"

# Combined policy table: index i = masked E.POLICIES[i]; index P0 + i = the
# same rule with unmasked `hybrid` as the retrieval arm.
CANDS = {
    "naive_masked": [0, 1],
    "audit_masked": list(range(P0)),
    "naive_unmasked": [P0, P0 + 1],
    "audit_unmasked": list(range(P0, 2 * P0)),
}
# score columns: (name, selection procedure, scored on masked outputs?)
SCORES = [
    ("naive_masked", "naive_masked", False),
    ("audit_masked", "audit_masked", False),
    ("naive_unmasked", "naive_unmasked", False),
    ("audit_unmasked", "audit_unmasked", False),
    ("naive_unmasked_deployed_masked", "naive_unmasked", True),
    ("audit_unmasked_deployed_masked", "audit_unmasked", True),
]
CONTRASTS = {
    "primary_audit_minus_naive": ("audit_masked", "naive_masked"),
    "L2a_naive_masked_minus_unmasked": ("naive_masked", "naive_unmasked"),
    "L2a_audit_masked_minus_unmasked": ("audit_masked", "audit_unmasked"),
    "L2b_naive_select_masked_minus_select_unmasked": ("naive_masked", "naive_unmasked_deployed_masked"),
    "L2b_audit_select_masked_minus_select_unmasked": ("audit_masked", "audit_unmasked_deployed_masked"),
}
# held-out action disagreement pairs (selection procedures)
ACTION_PAIRS = {
    "audit_vs_naive": ("audit_masked", "naive_masked"),
    "audit_masked_vs_unmasked": ("audit_masked", "audit_unmasked"),
    "naive_masked_vs_unmasked": ("naive_masked", "naive_unmasked"),
}


# --------------------------------------------------------------------------
class Gen:
    """Per-generator exact and float quiz-level tables."""

    def __init__(self, model, questions, qids, quizzes, zi):
        rows = {c: C.load_rows(model, c) for c in ("closed_book", "hybrid", "hybrid_masked")}
        for c, r in rows.items():
            missing = [q for q in qids if q not in r]
            if missing:
                raise AssertionError(f"{model}/{c}: {len(missing)} questions missing")
        raw = {c: C.f1_scores(rows[c], questions) for c in rows}
        ex = {c: [E.exact_f1(raw[c][q]) for q in qids] for c in rows}
        dates = [E.date_of(questions[q]) for q in qids]
        static = np.array([not C.is_fresh(questions[q]) for q in qids])
        Q, Z, P = len(qids), len(quizzes), 2 * P0
        act = np.array([[E.retrieves(p, d) for d in dates] for p in E.POLICIES])
        FX = [[None] * Q for _ in range(P)]
        for a, arm in enumerate(ARMS):
            for i in range(P0):
                for k in range(Q):
                    cond = RAG_COND[arm] if act[i, k] else "closed_book"
                    FX[a * P0 + i][k] = ex[cond][k]
        self.model = model
        self.FQ = np.array([[float(x) for x in row] for row in FX])          # (P, Q)
        members = [np.flatnonzero(zi == z) for z in range(Z)]
        st_members = [m[static[m]] for m in members]
        fr_members = [m[~static[m]] for m in members]
        self.n_z = np.array([len(m) for m in members], float)
        self.ns_z = np.array([len(m) for m in st_members], float)
        self.nf_z = np.array([len(m) for m in fr_members], float)

        def fsum(j, m):
            return sum((FX[j][k] for k in m), Fraction(0))

        self.QMF = [[fsum(j, members[z]) / len(members[z]) for z in range(Z)] for j in range(P)]
        self.QM = np.array([[float(x) for x in row] for row in self.QMF])
        self.QS = np.array([[float(fsum(j, members[z])) for z in range(Z)] for j in range(P)])
        ss = [[fsum(j, st_members[z]) for z in range(Z)] for j in range(P)]
        fs = [[fsum(j, fr_members[z]) for z in range(Z)] for j in range(P)]
        self.SS = np.array([[float(x) for x in row] for row in ss])
        self.FS = np.array([[float(x) for x in row] for row in fs])
        self.SM = np.array([[float(ss[j][z] / int(self.ns_z[z])) if self.ns_z[z] else 0.0
                             for z in range(Z)] for j in range(P)])
        self.FM = np.array([[float(fs[j][z] / int(self.nf_z[z])) if self.nf_z[z] else 0.0
                             for z in range(Z)] for j in range(P)])
        self.EQ = np.array([[[self.QMF[i][z] == self.QMF[j][z] for z in range(Z)]
                             for j in range(P)] for i in range(P)])            # (P, P, Z)
        self.CALLS = np.array([[act[i, m].sum() for m in members] for i in range(P0)], float)
        self.DQ = np.array([[[(act[i, m] != act[j, m]).sum() for m in members]
                             for j in range(P0)] for i in range(P0)], float)   # (P0, P0, Z)
        self.static = static
        self.FX = FX
        self.zi = zi


# --------------------------------------------------------------------------
def select_vec(D: Gen, Wdev: np.ndarray, cand: list[int], stats: dict) -> np.ndarray:
    """E5 selection for every row of Wdev (dev multiplicities per quiz):
    maximise the multiplicity-weighted sum of quiz means; exact ties go to the
    earliest candidate. Floats rank; near-ties are resolved exactly."""
    cand = np.asarray(cand)
    V = Wdev @ D.QM[cand].T                                  # (N, K)
    best = V.max(1)
    near = V >= (best - NEAR)[:, None]
    choice = near.argmax(1)
    multi = np.flatnonzero(near.sum(1) > 1)
    if multi.size:
        K = len(cand)
        iu, ju = np.triu_indices(K, 1)
        neq = (~D.EQ[cand[iu], cand[ju]]).astype(np.float64)   # (pairs, Z)
        differ = (Wdev[multi] @ neq.T) > 0                   # pair differs on a dev quiz
        nm = near[multi]
        needs = (nm[:, iu] & nm[:, ju] & differ).any(1)
        stats["exact_tie_structural"] += int((~needs).sum())
        for row in multi[needs]:
            stats["exact_fallback"] += 1
            w = np.rint(Wdev[row]).astype(np.int64)
            nz = np.flatnonzero(w)
            ks = np.flatnonzero(near[row])
            vals = [sum((int(w[z]) * D.QMF[cand[k]][z] for z in nz), Fraction(0)) for k in ks]
            bk, bv = ks[0], vals[0]
            for k, v in zip(ks[1:], vals[1:]):
                if v > bv:
                    bk, bv = k, v
            choice[row] = bk
            stats["exact_tie_after_fallback"] += int(sum(v == bv for v in vals) > 1)
    return cand[choice]


def crossfit(D: Gen, M: np.ndarray, fold: np.ndarray, chunk: int = 50,
             keep_choices: bool = False) -> dict:
    """Run the E5 cross-fitting on every row of M (quiz multiplicities, (N0, Z))
    with fold labels fold ((N0, R, Z), fold unit = distinct quiz). Returns
    per-row statistics averaged over repeats, as arrays of length N0."""
    N0, R, Z = fold.shape
    ks = np.arange(F)
    out = defaultdict(list)
    stats = defaultdict(int)
    for s in range(0, N0, chunk):
        Mc, fc = M[s:s + chunk], fold[s:s + chunk]
        n0 = len(Mc)
        onehot = fc[:, :, None, :] == ks[None, None, :, None]           # (n0,R,F,Z)
        Wt = (onehot * Mc[:, None, None, :]).reshape(-1, Z)
        Wd = np.repeat(Mc, R * F, axis=0) - Wt
        rows = np.arange(len(Wt))
        ch = {k: select_vec(D, Wd, c, stats) for k, c in CANDS.items()}
        if keep_choices:
            for k, v in ch.items():
                out["choice/" + k].append(v.reshape(n0, R, F))

        def agg(v):
            return v.reshape(n0, R * F).sum(1) / R

        den = {"qw": Mc.sum(1), "pooled": Mc @ D.n_z,
               "static_qw": Mc @ (D.ns_z > 0), "static_pooled": Mc @ D.ns_z,
               "fresh_qw": Mc @ (D.nf_z > 0), "fresh_pooled": Mc @ D.nf_z}
        T = {"qw": Wt @ D.QM.T, "pooled": Wt @ D.QS.T,
             "static_qw": Wt @ D.SM.T, "static_pooled": Wt @ D.SS.T,
             "fresh_qw": Wt @ D.FM.T, "fresh_pooled": Wt @ D.FS.T}
        for name, proc, deploy_masked in SCORES:
            idx = ch[proc] - P0 if deploy_masked else ch[proc]
            for w in T:
                out[f"{w}/{name}"].append(agg(T[w][rows, idx]) / den[w])
        cb = 0
        for w in ("static_qw", "static_pooled"):
            out[f"{w}/CB"].append(agg(T[w][:, cb]) / den[w])
        calls = Wt @ D.CALLS.T
        for proc in CANDS:
            out[f"call_rate/{proc}"].append(agg(calls[rows, ch[proc] % P0]) / den["pooled"])
        dq =Wt @ D.DQ.reshape(P0 * P0, Z).T
        for pname, (a, b) in ACTION_PAIRS.items():
            v = dq[rows, (ch[a] % P0) * P0 + (ch[b] % P0)]
            out[f"disagree_rate/{pname}"].append(agg(v) / den["pooled"])
            out[f"label_differs_fits/{pname}"].append(
                ((ch[a] % P0) != (ch[b] % P0)).reshape(n0, R * F).sum(1))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["_stats"] = dict(stats)
    return res


def folds_from_keys(M: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Distinct quizzes present in each draw, ordered by random keys, cut into F
    contiguous near-equal groups (sizes as in E.folds_for). Absent quizzes get a
    fold label but zero weight."""
    B, R, Z = keys.shape
    present = M > 0
    k = np.where(present[:, None, :], keys, np.inf)
    order = np.argsort(k, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(Z), order.shape).copy(), axis=-1)
    u = present.sum(1)
    sizes = u[:, None] // F + (np.arange(F)[None, :] < (u % F)[:, None])
    bounds = np.cumsum(sizes, 1)[:, :F - 1]
    return (rank[..., None] >= bounds[:, None, None, :]).sum(-1)


def original_folds(quizzes: list[str], R: int) -> np.ndarray:
    pos = {z: i for i, z in enumerate(quizzes)}
    fold = np.zeros((1, R, len(quizzes)), dtype=np.int64)
    for r in range(R):
        for k, fz in enumerate(E.folds_for(quizzes, r)):
            for z in fz:
                fold[0, r, pos[z]] = k
    return fold


def per_question_crossfit(D: Gen, res: dict, fold: np.ndarray, zi: np.ndarray) -> dict:
    """Per-question cross-fitted F1 (mean over repeats) from original-data choices."""
    R = fold.shape[1]
    foldq = fold[0][:, zi]                                            # (R, Q)
    Q = len(zi)
    out = {}
    for name, proc, deploy_masked in SCORES:
        ch = res["choice/" + proc][0]                                   # (R, F)
        idx = np.take_along_axis(ch, foldq, axis=1)                     # (R, Q)
        if deploy_masked:
            idx = idx - P0
        out[name] = D.FQ[idx, np.arange(Q)[None, :]].mean(0)
    return out


def snap(d: np.ndarray, exact_zero: np.ndarray | None = None) -> np.ndarray:
    d = np.where(np.abs(d) < ZERO, 0.0, d)
    if exact_zero is not None:
        d = np.where(exact_zero, 0.0, d)
    return d


def summ(obs, draws, nd=2):
    return C.summarize(float(obs), draws, nd=nd)


# --------------------------------------------------------------------------
def verdict(unchanged: bool, p_holm: float | None, est: float) -> str:
    if unchanged:
        return "unchanged actions"
    if p_holm is not None and p_holm < ALPHA:
        return "significant gain" if est > 0 else "significant loss"
    return "unresolved"


def reference_draw(D: Gen, m_row: np.ndarray, fold_row: np.ndarray) -> dict:
    """Slow, explicit reference for one outer draw: expand the resampled data
    set into pseudo-quizzes (quiz copies) and per-question rows, check the fold
    properties, select with exact Fraction sums (first strict maximum in
    candidate order, as E.select), and aggregate per-question cross-fitted
    scores. Used to verify the vectorised engine on a few draws."""
    R, Z = fold_row.shape
    pseudo = [(z, c) for z in range(Z) for c in range(int(m_row[z]))]
    present = sorted({z for z, _ in pseudo})
    for r in range(R):
        sizes = [sum(1 for z in present if fold_row[r, z] == k) for k in range(F)]
        assert max(sizes) - min(sizes) <= 1, sizes
    members = {z: np.flatnonzero(D.zi == z) for z in present}
    pm = {}

    def pmean(p, z):
        if (p, z) not in pm:
            pm[(p, z)] = sum((D.FX[p][k] for k in members[z]), Fraction(0)) / len(members[z])
        return pm[(p, z)]

    acc = {name: defaultdict(float) for name, _, _ in SCORES}
    for r in range(R):
        for k in range(F):
            dev = [z for z, c in pseudo if fold_row[r, z] != k]
            test = [(z, c) for z, c in pseudo if fold_row[r, z] == k]
            chosen = {}
            for proc, cand in CANDS.items():
                best, bv = None, None
                for p in cand:
                    v = sum((pmean(p, z) for z in dev), Fraction(0))
                    if bv is None or v > bv:
                        best, bv = p, v
                chosen[proc] = best
            for z, c in test:
                for kq in members[z]:
                    for name, proc, dep in SCORES:
                        p = chosen[proc] - P0 if dep else chosen[proc]
                        acc[name][(z, c, int(kq))] += D.FQ[p, kq]
    out = {}
    for name in acc:
        cfv = {key: v / R for key, v in acc[name].items()}
        byq = defaultdict(list)
        for (z, c, _), v in cfv.items():
            byq[(z, c)].append(v)
        out[f"qw/{name}"] = float(np.mean([np.mean(v) for v in byq.values()]))
        out[f"pooled/{name}"] = float(np.mean(list(cfv.values())))
    return out


def analyse(model, D, qids, quizzes, zi, stored, boot, M_out, keys, R_out, sens_R, n_ref):
    t0 = time.perf_counter()
    g = stored["generators"][model]
    rec = {"model": model}

    # ---- validation: 20-repeat original data reproduces stored E5 -----------
    fold20 = original_folds(quizzes, E.REPEATS)
    r20 = crossfit(D, np.ones((1, len(quizzes))), fold20, keep_choices=True)
    lab = {"naive_masked": lambda i: E.POLICIES[i], "audit_masked": lambda i: E.POLICIES[i],
           "naive_unmasked": lambda i: "CB" if i == P0 else "RAG_UNMASKED"}
    mism = 0
    for fit in g["fits"]:
        r, k = fit["repeat"], fit["fold"]
        for proc, key in (("naive_masked", "naive"), ("audit_masked", "audit"),
                          ("naive_unmasked", "naive_unmasked")):
            mism += lab[proc](int(r20["choice/" + proc][0, r, k])) != fit[key]
    cf20 = per_question_crossfit(D, r20, fold20, zi)
    allm = np.ones(len(qids), bool)
    chk = {
        "fits_matching_stored": f"{3 * len(g['fits']) - mism}/{3 * len(g['fits'])}",
        "primary_qw_stored": g["primary"]["quiz_weighted"]["estimate"],
        "primary_qw_recomputed": round(float(r20["qw/audit_masked"][0] - r20["qw/naive_masked"][0]), 2),
        "primary_pooled_stored": g["primary"]["question_pooled"]["estimate"],
        "primary_pooled_recomputed": round(float(r20["pooled/audit_masked"][0] - r20["pooled/naive_masked"][0]), 2),
        "heldout_action_disagreement_stored": g["cross_fitted"]["heldout_action_disagreement"]["n"],
        "heldout_action_disagreement_rate_recomputed": round(float(r20["disagree_rate/audit_vs_naive"][0]), 4),
        "call_rate_stored": g["cross_fitted"]["retrieval_call_rate"],
        "call_rate_recomputed": {"naive": round(float(r20["call_rate/naive_masked"][0]), 3),
                                 "audit": round(float(r20["call_rate/audit_masked"][0]), 3)},
        "leakage_naive_qw_stored": g["leakage_variant"]["masked_minus_unmasked"]["quiz_weighted"]["estimate"],
        "leakage_naive_qw_recomputed": round(float(r20["qw/naive_masked"][0] - r20["qw/naive_unmasked"][0]), 2),
    }
    stored_ci = g["primary"]["quiz_weighted"]["ci95"]
    re_ci = C.summarize(*boot.quizw(cf20["audit_masked"] - cf20["naive_masked"], allm))["ci95"]
    chk["primary_qw_ci_stored"], chk["primary_qw_ci_recomputed"] = stored_ci, re_ci
    ok = (mism == 0 and chk["primary_qw_stored"] == chk["primary_qw_recomputed"]
          and chk["primary_pooled_stored"] == chk["primary_pooled_recomputed"]
          and stored_ci == re_ci
          and chk["leakage_naive_qw_stored"] == chk["leakage_naive_qw_recomputed"]
          and chk["call_rate_stored"] == chk["call_rate_recomputed"])
    chk["all_match"] = bool(ok)
    chk["selection_tie_stats_20rep"] = r20["_stats"]
    rec["validation_vs_stored"] = chk
    if not ok:
        raise AssertionError(f"{model}: re-implementation does not reproduce stored E5: {chk}")
    unchanged_stored = g["cross_fitted"]["heldout_action_disagreement"]["n"] == 0

    # ---- stored-style conditional CIs for the new L2 contrasts (20 repeats) --
    cond = {}
    for cname, (a, b) in CONTRASTS.items():
        d = cf20[a] - cf20[b]
        cond[cname] = {w: C.summarize(*boot.est(w, d, allm)) for w in C.WEIGHTINGS}
    rec["conditional_20rep"] = cond
    rec["_cond_p_qw"] = {c: cond[c]["quiz_weighted"]["p_two_sided_boot"] for c in cond}

    # ---- original data, R_out repeats (point estimates for the refit) -------
    fold_o = original_folds(quizzes, R_out)
    ro = crossfit(D, np.ones((1, len(quizzes))), fold_o, keep_choices=True)

    # ---- outer bootstrap ----------------------------------------------------
    t1 = time.perf_counter()
    fold_b = folds_from_keys(M_out, keys[:, :R_out])
    rb = crossfit(D, M_out, fold_b)
    t_outer = time.perf_counter() - t1
    ref_ok, ref_maxdiff = 0, 0.0
    for b in range(n_ref):
        ref = reference_draw(D, M_out[b], fold_b[b])
        diffs = [abs(ref[k] - rb[k][b]) for k in ref]
        ref_maxdiff = max(ref_maxdiff, max(diffs))
        ref_ok += max(diffs) < 1e-9
    if ref_ok != n_ref:
        raise AssertionError(f"{model}: vectorised refit disagrees with reference ({ref_maxdiff})")
    rec["reference_check"] = {"draws_checked": n_ref, "draws_matching": ref_ok,
                              "max_abs_diff": ref_maxdiff,
                              "note": ("explicit pseudo-quiz expansion with exact Fraction "
                                       "selection reproduces every score column of the "
                                       "vectorised engine on these draws")}

    def refit_block(res_o, res_b, R):
        blk = {"repeats": R}
        same_b = res_b["disagree_rate/audit_vs_naive"] == 0
        same_o = bool(res_o["disagree_rate/audit_vs_naive"][0] == 0)
        for cname, (a, b) in CONTRASTS.items():
            blk[cname] = {}
            for w in ("qw", "pooled"):
                ez = same_b if cname.startswith("primary") else None
                dr = snap(res_b[f"{w}/{a}"] - res_b[f"{w}/{b}"], ez)
                ob = float(snap(res_o[f"{w}/{a}"] - res_o[f"{w}/{b}"],
                                np.array([same_o]) if cname.startswith("primary") else None)[0])
                blk[cname]["quiz_weighted" if w == "qw" else "question_pooled"] = summ(ob, dr)
        sub = {}
        for grp in ("static", "fresh"):
            sub[grp] = {}
            for w in ("qw", "pooled"):
                key = "quiz_weighted" if w == "qw" else "question_pooled"
                dr = snap(res_b[f"{grp}_{w}/audit_masked"] - res_b[f"{grp}_{w}/naive_masked"], same_b)
                ob = snap(res_o[f"{grp}_{w}/audit_masked"] - res_o[f"{grp}_{w}/naive_masked"],
                          np.array([same_o]))[0]
                sub[grp].setdefault("audit_minus_naive", {})[key] = summ(ob, dr)
                if grp == "static":
                    for proc, lbl in (("audit_masked", "audit_minus_CB"), ("naive_masked", "naive_minus_CB")):
                        dr = snap(res_b[f"static_{w}/{proc}"] - res_b[f"static_{w}/CB"])
                        ob = snap(res_o[f"static_{w}/{proc}"] - res_o[f"static_{w}/CB"])[0]
                        sub[grp].setdefault(lbl, {})[key] = summ(ob, dr)
        blk["subgroups"] = sub
        blk["call_rate"] = {}
        for proc in ("naive_masked", "audit_masked"):
            blk["call_rate"][proc] = summ(res_o[f"call_rate/{proc}"][0], res_b[f"call_rate/{proc}"], nd=3)
        dcr = res_b["call_rate/audit_masked"] - res_b["call_rate/naive_masked"]
        blk["call_rate"]["audit_minus_naive"] = summ(
            res_o["call_rate/audit_masked"][0] - res_o["call_rate/naive_masked"][0], dcr, nd=3)
        blk["actions"] = {
            "original_heldout_action_disagreement_rate": round(float(res_o["disagree_rate/audit_vs_naive"][0]), 4),
            "share_draws_identical_actions_audit_vs_naive": round(float(same_b.mean()), 4),
            "mean_draw_action_disagreement_rate_audit_vs_naive": round(float(res_b["disagree_rate/audit_vs_naive"].mean()), 4),
            "share_draws_identical_actions_audit_masked_vs_unmasked": round(float((res_b["disagree_rate/audit_masked_vs_unmasked"] == 0).mean()), 4),
            "share_draws_identical_actions_naive_masked_vs_unmasked": round(float((res_b["disagree_rate/naive_masked_vs_unmasked"] == 0).mean()), 4),
        }
        blk["_draws_primary_qw"] = snap(res_b["qw/audit_masked"] - res_b["qw/naive_masked"], same_b)
        blk["primary_audit_minus_naive"]["quiz_weighted"]["draw_mean"] = round(float(blk["_draws_primary_qw"].mean()), 2)
        blk["primary_audit_minus_naive"]["quiz_weighted"]["draw_median"] = round(float(np.median(blk["_draws_primary_qw"])), 2)
        blk["selection_tie_stats"] = res_b["_stats"]
        return blk

    rec["refit"] = refit_block(ro, rb, R_out)
    rec["refit"]["original_data_selection_counts"] = {
        proc: {lab_: int(n) for lab_, n in zip(*np.unique(
            [E.POLICIES[int(i) % P0] + ("" if int(i) < P0 else "|unmasked")
             for i in ro["choice/" + proc].ravel()], return_counts=True))}
        for proc in CANDS}
    rec["refit"]["original_data_action_disagreement_audit_masked_vs_unmasked"] = round(
        float(ro["disagree_rate/audit_masked_vs_unmasked"][0]), 4)
    if sens_R:
        fold_s = original_folds(quizzes, sens_R)
        rs_o = crossfit(D, np.ones((1, len(quizzes))), fold_s)
        rs_b = crossfit(D, M_out, folds_from_keys(M_out, keys[:, :sens_R]))
        blk = refit_block(rs_o, rs_b, sens_R)
        rec["refit_sensitivity"] = {k: blk[k] for k in
                                    ("repeats", "primary_audit_minus_naive", "actions", "_draws_primary_qw")}

    # ---- stored (conditional) summaries pulled from e5_decision.json -------
    cf = g["cross_fitted"]
    rec["stored"] = {
        "selection_counts": {k: {p: v["n"] for p, v in cf["selection_counts"][k].items()}
                             for k in ("naive", "audit", "naive_unmasked")},
        "n_fits": cf["n_fits"],
        "primary": {w: g["primary"][w] for w in C.WEIGHTINGS},
        "p_holm": g["primary"]["p_holm"],
        "heldout_action_disagreement": cf["heldout_action_disagreement"],
        "unchanged_actions": unchanged_stored,
        "call_rate": cf["retrieval_call_rate"],
        "test_f1_qw": {k: cf["test_f1"][k]["quiz_weighted"]["estimate"] for k in cf["test_f1"]},
        "subgroups": {grp: {"paired": g["subgroups"][grp]["paired"],
                            "call_rate": g["subgroups"][grp]["retrieval_call_rate"],
                            "n": {"questions": g["subgroups"][grp]["n_questions"],
                                  "quizzes": g["subgroups"][grp]["n_quizzes"]}}
                      for grp in ("static", "fresh")},
        "leakage_variant_naive": g["leakage_variant"]["masked_minus_unmasked"],
        "leakage_fits_flipping": g["leakage_variant"]["fits_where_CB_vs_RAG_choice_differs"],
        "verdict_e5_wording": g["verdict"],
    }
    rec["audit_selection_counts_20rep"] = {
        "masked": g["cross_fitted"]["selection_counts"]["audit"],
        "unmasked": dict(zip(*[x.tolist() for x in np.unique(
            [E.POLICIES[int(i) - P0] for i in r20["choice/audit_unmasked"].ravel()],
            return_counts=True)])),
        "fits_label_differs": int(r20["label_differs_fits/audit_masked_vs_unmasked"][0]),
        "heldout_action_disagreement_rate": round(float(r20["disagree_rate/audit_masked_vs_unmasked"][0]), 4),
    }
    rec["runtime_s"] = {"total": round(time.perf_counter() - t0, 2), "outer_bootstrap": round(t_outer, 4)}
    return rec


# --------------------------------------------------------------------------
def fci(d, nd=2):
    if d.get("ci95") is None:
        return f"{d['estimate']:+.{nd}f} [n/a]"
    lo, hi = d["ci95"]
    return f"{d['estimate']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def counts_str(c: dict, n: int) -> str:
    return ", ".join(f"{p} {v}/{n}" for p, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def write_md(out: dict, path: Path) -> None:
    G = out["generators"]
    R = out["design"]["outer_bootstrap"]["repeats"]
    B = out["design"]["outer_bootstrap"]["B"]
    L = ["# R4: E5 date-policy selection study, symmetric table with selection-refit inference", "",
         "Exploratory, internally cross-validated date-policy selection study on cached outputs "
         "(878 questions, 199 weekly quizzes, 9 generators). Internal validation over random weekly "
         "quizzes of an already-inspected dataset; not a prospective or independent temporal test. "
         "Token F1, 0-100. Naive = choose CB vs RAG; audit = choose among CB, RAG and 8 date boundaries. "
         "The comparison enlarges the policy class with question date (L1); it does not isolate L2/L3.", "",
         "**Stored** = `e5_decision.json`: 20 x 5-fold cross-fitting, paired quiz bootstrap (B=5000) "
         "of already cross-fitted per-question scores, i.e. conditional on the fitted selections. "
         f"**Refit** = outer grouped bootstrap (B={B}, seed {SEED_OUTER}) that resamples weekly quizzes "
         f"and reruns the whole {R} x 5-fold cross-fitting and selection in every draw (copies of a quiz share a fold); "
         f"refit point estimate = original data with {R} repeats. Holm across the 9 generators on the "
         "quiz-weighted p in both cases.", "",
         "Verdicts (symmetric): *unchanged actions* = audit and naive take identical held-out retrieval "
         "actions in all 100 stored fits; otherwise *significant gain* / *significant loss* if Holm p < 0.05; "
         "otherwise *unresolved*.", "",
         "## 1. Primary: audit - naive, quiz-weighted", "",
         "| generator | naive picks (100 fits) | audit picks (100 fits) | stored est. [95% CI] | stored Holm p | stored verdict "
         f"| refit est. ({R} rep) [95% CI] | refit p | refit Holm p | refit verdict | draws with identical actions "
         "| refit 20-rep sensitivity est. [95% CI], Holm p |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, rec in G.items():
        st, rf = rec["stored"], rec["refit"]
        L.append(" | ".join([
            f"| {m}", counts_str(st["selection_counts"]["naive"], st["n_fits"]),
            counts_str(st["selection_counts"]["audit"], st["n_fits"]),
            fci(st["primary"]["quiz_weighted"]), f"{st['p_holm']:.4f}", rec["verdict_stored"],
            fci(rf["primary_audit_minus_naive"]["quiz_weighted"]),
            f"{rf['primary_audit_minus_naive']['quiz_weighted']['p_two_sided_boot']:.4f}",
            f"{rf['p_holm']:.4f}", rec["verdict_refit"],
            f"{rf['actions']['share_draws_identical_actions_audit_vs_naive']:.3f}",
            (f"{fci(rec['refit_sensitivity']['primary_audit_minus_naive']['quiz_weighted'])}, "
             f"{rec['refit_sensitivity']['p_holm']:.4f} |") if "refit_sensitivity" in rec else "n/a |"]))
    L += ["", "Refit draws are centred below the original-data estimates where selection matters (mean of draws: "
          + ", ".join(f"{m} {rec['refit']['primary_audit_minus_naive']['quiz_weighted']['draw_mean']:+.2f}"
                      for m, rec in G.items() if not rec["stored"]["unchanged_actions"])
          + "): a resampled data set holds about 63% distinct quizzes, so selection in a draw is noisier than "
          "in the original data and the refit interval is, if anything, conservative for the original procedure.",
          "", "Question-pooled audit - naive (secondary weighting):", "",
          "| generator | stored est. [95% CI] | stored p | refit est. [95% CI] | refit p |", "|---|---|---|---|---|"]
    for m, rec in G.items():
        sp = rec["stored"]["primary"]["question_pooled"]
        rp = rec["refit"]["primary_audit_minus_naive"]["question_pooled"]
        L.append(f"| {m} | {fci(sp)} | {sp['p_two_sided_boot']:.4f} | {fci(rp)} | {rp['p_two_sided_boot']:.4f} |")
    S = out["summary"]
    L += ["", "## Summary counts", "",
          f"- Stored conditional inference: {S['stored']['significant gain']} significant gains, "
          f"{S['stored']['significant loss']} significant losses, {S['stored']['unresolved']} unresolved, "
          f"{S['stored']['unchanged actions']} unchanged actions.",
          f"- Selection-refit inference: {S['refit']['significant gain']} significant gains, "
          f"{S['refit']['significant loss']} significant losses, {S['refit']['unresolved']} unresolved, "
          f"{S['refit']['unchanged actions']} unchanged actions.",
          f"- Verdicts changed by refit inference: {'; '.join(S['verdicts_changed']) or 'none'}."]
    if "refit_sensitivity" in next(iter(G.values())):
        L.append(f"- Sensitivity, refit with {S['sensitivity_repeats']} repeats per draw: "
                 f"{S['refit_sensitivity']['significant gain']} gains, {S['refit_sensitivity']['significant loss']} losses, "
                 f"{S['refit_sensitivity']['unresolved']} unresolved, {S['refit_sensitivity']['unchanged actions']} unchanged.")
    L += ["", "## 2. Cost and subgroups (audit - naive unless stated; quiz-weighted)", "",
          "Call rate = share of questions answered with retrieval (cross-fitted, stored). "
          "Static = year < 2025 (404 q, 125 quizzes); Fresh = year >= 2025 (474 q, 74 quizzes). "
          "Static harm = Static F1 of the procedure minus always-closed-book on Static.", "",
          "| generator | call rate naive | call rate audit | Static audit-naive stored [CI] | refit [CI] "
          "| Fresh audit-naive stored [CI] | refit [CI] | Static naive-CB stored [CI] | Static audit-CB stored [CI] | refit [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rec in G.items():
        st, rf = rec["stored"], rec["refit"]["subgroups"]
        ss, sf = st["subgroups"]["static"]["paired"], st["subgroups"]["fresh"]["paired"]
        L.append(" | ".join([
            f"| {m}", f"{st['call_rate']['naive']:.3f}", f"{st['call_rate']['audit']:.3f}",
            fci(ss["audit_minus_naive"]["quiz_weighted"]), fci(rf["static"]["audit_minus_naive"]["quiz_weighted"]),
            fci(sf["audit_minus_naive"]["quiz_weighted"]), fci(rf["fresh"]["audit_minus_naive"]["quiz_weighted"]),
            fci(ss["naive_minus_CB"]["quiz_weighted"]), fci(ss["audit_minus_CB"]["quiz_weighted"]),
            fci(rf["static"]["audit_minus_CB"]["quiz_weighted"]) + " |"]))
    L += ["", "## 3. Incremental value of L2 masking within the date-conditioned class (secondary)", "",
          "(a) own-arm: procedure run and scored on masked RAG outputs minus run and scored on unmasked "
          "outputs (not a deployable contrast: unmasked outputs may use post-question evidence). "
          "(b) decision value: policy selected on masked minus policy selected on unmasked outputs, both "
          "scored on masked held-out outputs; 0 when masking leaves held-out actions unchanged. "
          "Conditional = stored-style (20 rep, B=5000); refit as above. Holm within each contrast across 9 generators "
          "(conditional p).", "",
          "| generator | audit picks masked | audit picks unmasked | held-out action disagreement* | (a) audit cond. [CI] | (a) refit [CI] "
          "| (a) Holm | (b) audit cond. [CI] | (b) refit [CI] | (b) Holm | (b) naive cond. [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, rec in G.items():
        sc = rec["audit_selection_counts_20rep"]
        cond, rf = rec["conditional_20rep"], rec["refit"]
        a, b = "L2a_audit_masked_minus_unmasked", "L2b_audit_select_masked_minus_select_unmasked"
        nb = "L2b_naive_select_masked_minus_select_unmasked"
        L.append(" | ".join([
            f"| {m}", counts_str({k: v["n"] for k, v in sc["masked"].items()}, 100),
            counts_str(sc["unmasked"], 100),
            f"{sc['heldout_action_disagreement_rate']:.4f}",
            fci(cond[a]["quiz_weighted"]), fci(rf[a]["quiz_weighted"]),
            f"{cond[a]['quiz_weighted']['p_holm_within_contrast']:.4f}",
            fci(cond[b]["quiz_weighted"]), fci(rf[b]["quiz_weighted"]),
            f"{cond[b]['quiz_weighted']['p_holm_within_contrast']:.4f}",
            fci(cond[nb]["quiz_weighted"]) + " |"]))
    L += ["", "*Share of held-out (question, repeat) instances in which the audit procedure's masked-selected and "
          "unmasked-selected policies differ in whether retrieval is used (20 repeats).", "",
          "## Runtime and checks", "",
          f"- Re-implementation reproduces all 300 stored selections per generator (naive, audit, naive-unmasked), "
          f"stored point estimates, stored quiz-weighted CI, call rates and leakage estimate: "
          f"{all(r['validation_vs_stored']['all_match'] for r in G.values())}.",
          f"- Slow explicit reference (pseudo-quiz expansion, exact Fraction selection) reproduces the vectorised "
          f"refit engine on the first {next(iter(G.values()))['reference_check']['draws_checked']} draws of every generator: "
          f"{all(r['reference_check']['draws_matching'] == r['reference_check']['draws_checked'] for r in G.values())}.",
          f"- Runtime (CPU, numpy): timing run with 20 draws took {out['runtime']['per_20_draw_estimate_s']} s wall "
          f"for the whole script (all 9 generators, including validation); the full run took "
          f"{out['runtime']['total_s']} s, of which the outer bootstrap took "
          f"{sum(out['runtime']['outer_bootstrap_s_per_generator']):.2f} s over 9 generators. B not reduced.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-B", type=int, default=B_OUTER)
    ap.add_argument("--repeats", type=int, default=R_OUTER)
    ap.add_argument("--sensitivity-repeats", type=int, default=0,
                    help="also rerun the outer bootstrap with this many repeats (0 = off)")
    ap.add_argument("--out-dir", type=Path, default=C.OUT)
    ap.add_argument("--timing-s-per-20", type=float, default=None,
                    help="runtime of a 20-draw timing run, recorded in the output")
    ap.add_argument("--models", nargs="*", default=E.GENERATORS)
    ap.add_argument("--reference-draws", type=int, default=5,
                    help="outer draws per generator re-checked with the slow reference")
    args = ap.parse_args()
    if args.B < 500 and args.out_dir == C.OUT:
        raise SystemExit("B below 500 only for timing runs (use --out-dir)")
    t_all = time.perf_counter()

    questions = C.load_questions()
    qids = sorted(questions)
    assert len(qids) == E.N_EXPECTED
    quizzes = sorted({C.quiz_id(q) for q in qids})
    Z = len(quizzes)
    zpos = {z: i for i, z in enumerate(quizzes)}
    zi = np.array([zpos[C.quiz_id(q)] for q in qids])
    stored = json.loads(STORED.read_text(encoding="utf-8"))
    boot = C.QuizBoot(qids, B=C.B_DEFAULT, seed=E.SEED)

    rng = np.random.default_rng(SEED_OUTER)
    idx = rng.integers(0, Z, size=(args.B, Z))
    M_out = np.zeros((args.B, Z))
    np.add.at(M_out, (np.repeat(np.arange(args.B), Z), idx.ravel()), 1.0)
    keys = rng.random((args.B, max(R_KEYS, args.sensitivity_repeats), Z))

    out = {
        "experiment": "R4 on E5: selection-refit outer bootstrap, symmetric table, L2 incremental value",
        "source": "results/analysis/e5_decision.json; pipeline/analysis/e5_decision.py (imported, unchanged)",
        "scope": ("exploratory, internally cross-validated date-policy selection study; random held-out "
                  "weekly quizzes of an already-inspected dataset, not prospective or independent temporal "
                  "validation; compares a two-policy class with a date-conditioned class (L1), does not "
                  "isolate L2/L3 except in the secondary masking contrasts"),
        "design": {
            "policies_boundaries_selection_ties_endpoint": "imported from e5_decision.py unchanged",
            "outer_bootstrap": {
                "B": args.B, "seed": SEED_OUTER, "repeats": args.repeats, "folds": F,
                "resampling": "weekly quizzes with replacement (numpy default_rng(seed).integers(0, 199, (B, 199)))",
                "fold_unit": ("distinct original quiz: all copies of a resampled quiz share one fold; distinct "
                              "quizzes ordered by default_rng keys drawn after the resampling indices and cut "
                              "into 5 contiguous near-equal groups (E.folds_for size rule)"),
                "selection_in_draw": ("quiz-weighted dev mean over quiz copies (multiplicity-weighted); exact "
                                      "ties to the earliest candidate: float ranking, structural ties detected "
                                      "from exact per-quiz equality, remaining near-ties (<1e-6) re-evaluated "
                                      "with Fractions"),
                "statistic": "cross-fitted audit - naive difference in the resampled data set, quiz-weighted and question-pooled",
                "ci": "percentile 95%", "p": "two-sided plus-one bootstrap p (analysis_common.summarize)",
                "multiplicity": "Holm across 9 generators on the quiz-weighted refit p",
                "point_estimate": f"original data, repeats 0..{args.repeats - 1} of E.folds_for",
            },
            "verdict_rule": ("unchanged actions if audit and naive held-out actions identical in all 100 "
                             "stored fits; else significant gain/loss if Holm p < 0.05 (sign of the "
                             "stored 20-repeat estimate); else unresolved"),
            "L2": {k: v for k, v in CONTRASTS.items() if k.startswith("L2")},
        },
        "generators": {},
    }

    per_gen_outer = []
    for m in args.models:
        D = Gen(m, questions, qids, quizzes, zi)
        rec = analyse(m, D, qids, quizzes, zi, stored, boot, M_out, keys,
                      R_out=args.repeats, sens_R=args.sensitivity_repeats,
                      n_ref=args.reference_draws)
        out["generators"][m] = rec
        per_gen_outer.append(rec["runtime_s"]["outer_bootstrap"])
        print(f"{m}: {rec['runtime_s']} validation={rec['validation_vs_stored']['all_match']}", flush=True)

    ms = list(out["generators"])
    G = out["generators"]
    # Holm, refit (primary quiz-weighted) and sensitivity
    for key in ("refit", "refit_sensitivity"):
        if key not in G[ms[0]]:
            continue
        ps = [G[m][key]["primary_audit_minus_naive"]["quiz_weighted"]["p_two_sided_boot"] for m in ms]
        if any(p is None for p in ps):
            continue
        adj = holm(ps)
        order = sorted(range(len(ms)), key=lambda i: ps[i])
        for rank, i in enumerate(order):
            G[ms[i]][key]["p_holm"] = adj[i]
            G[ms[i]][key]["holm_step_ci_quiz_weighted"] = E.holm_step_ci(
                G[ms[i]][key]["_draws_primary_qw"], rank, len(ms))
        pq = [G[m][key]["primary_audit_minus_naive"]["question_pooled"]["p_two_sided_boot"] for m in ms]
        for m, a in zip(ms, holm(pq)):
            G[m][key]["p_holm_question_pooled_secondary"] = a
    # Holm within each L2 contrast family (conditional p), and refit p
    for cname in CONTRASTS:
        if not cname.startswith("L2"):
            continue
        ps = [G[m]["conditional_20rep"][cname]["quiz_weighted"]["p_two_sided_boot"] for m in ms]
        for m, a in zip(ms, holm(ps)):
            G[m]["conditional_20rep"][cname]["quiz_weighted"]["p_holm_within_contrast"] = a
        rps = [G[m]["refit"][cname]["quiz_weighted"]["p_two_sided_boot"] for m in ms]
        if all(p is not None for p in rps):
            for m, a in zip(ms, holm(rps)):
                G[m]["refit"][cname]["quiz_weighted"]["p_holm_within_contrast"] = a

    cats = ("significant gain", "significant loss", "unresolved", "unchanged actions")
    summary = {"stored": dict.fromkeys(cats, 0), "refit": dict.fromkeys(cats, 0)}
    changed = []
    sens = "refit_sensitivity" in G[ms[0]] and "p_holm" in G[ms[0]]["refit_sensitivity"]
    if sens:
        summary["refit_sensitivity"] = dict.fromkeys(cats, 0)
        summary["sensitivity_repeats"] = args.sensitivity_repeats
    for m in ms:
        rec = G[m]
        st = rec["stored"]
        est20 = st["primary"]["quiz_weighted"]["estimate"]
        rec["verdict_stored"] = verdict(st["unchanged_actions"], st["p_holm"], est20)
        est5 = rec["refit"]["primary_audit_minus_naive"]["quiz_weighted"]["estimate"]
        rec["refit"]["sign_agrees_with_stored_20rep"] = bool(np.sign(est5) == np.sign(est20))
        rec["verdict_refit"] = verdict(st["unchanged_actions"], rec["refit"].get("p_holm"), est20)
        summary["stored"][rec["verdict_stored"]] += 1
        summary["refit"][rec["verdict_refit"]] += 1
        if rec["verdict_stored"] != rec["verdict_refit"]:
            changed.append(f"{m}: {rec['verdict_stored']} -> {rec['verdict_refit']}")
        if sens:
            rec["verdict_refit_sensitivity"] = verdict(st["unchanged_actions"],
                                                       rec["refit_sensitivity"]["p_holm"], est20)
            summary["refit_sensitivity"][rec["verdict_refit_sensitivity"]] += 1
        for key in ("refit", "refit_sensitivity"):
            if key in rec:
                rec[key].pop("_draws_primary_qw", None)
        rec.pop("_cond_p_qw", None)
    summary["verdicts_changed"] = changed
    summary["sentence"] = (
        f"Under selection-refit inference: {summary['refit']['significant gain']} significant gains, "
        f"{summary['refit']['significant loss']} significant losses, {summary['refit']['unresolved']} unresolved, "
        f"{summary['refit']['unchanged actions']} unchanged actions (stored conditional inference: "
        f"{summary['stored']['significant gain']}/{summary['stored']['significant loss']}/"
        f"{summary['stored']['unresolved']}/{summary['stored']['unchanged actions']}).")
    out["summary"] = summary
    total = time.perf_counter() - t_all
    out["runtime"] = {
        "total_s": round(total, 1),
        "outer_bootstrap_s_per_generator": [round(x, 4) for x in per_gen_outer],
        "per_20_draw_estimate_s": args.timing_s_per_20,
        "B_reduced": False, "device": "CPU, numpy",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jp = args.out_dir / "r4_e5_outer.json"
    jp.write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n",
                  encoding="utf-8")
    if all(r["refit"]["primary_audit_minus_naive"]["quiz_weighted"]["ci95"] is not None for r in G.values()):
        write_md(out, args.out_dir / "r4_e5_table.md")
    print(summary["sentence"])
    print("changed:", changed or "none")
    print(f"wrote {jp} ({total:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
