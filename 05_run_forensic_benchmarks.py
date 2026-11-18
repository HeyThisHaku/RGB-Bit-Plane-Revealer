import argparse
import importlib.util
import io
import json
import math
import os
import platform
import random
import sys
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = Path(__file__).resolve().parent.parent
DATASETS = ["SSC_android_icon", "BCC_pokemon", "Gray", "FC_full_color"]
METHODS = ["LSB", "PVD", "BPCS"]
PAYLOADS = [
    "1_Plaintext_LoremIpsum",
    "2_Structured_Binary",
    "3_Compressed_Zlib",
    "4_Encrypted_AES256",
]

REGIMES = [
    ("JPEG_Q95", "jpeg", 95),
    ("JPEG_Q85", "jpeg", 85),
    ("JPEG_Q75", "jpeg", 75),
    ("AWGN_s1", "awgn", 1.0),
    ("AWGN_s3", "awgn", 3.0),
    ("AWGN_s5", "awgn", 5.0),
    ("Crop_8px", "crop", 8),
    ("Resize_0.9x", "resize", 0.9),
]


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


CODE_DIR = Path(__file__).resolve().parent
MOD_01 = _load_mod("mod_01", CODE_DIR / "01_build_unified_cohort.py")
MOD_02 = _load_mod("mod_02", CODE_DIR / "02_extract_components_and_mode1.py")
MOD_03 = _load_mod("mod_03", CODE_DIR / "03_evaluate_mode2_cv.py")
MOD_04 = _load_mod("mod_04", CODE_DIR / "04_extract_baselines_and_spam.py")


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> Dict[str, Any]:
    """Compute TP/FP/TN/FN, Acc, BalAcc, Prec, Rec, F1, FPR, MCC, AUC, and 95% Wilson CIs."""
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    tn = int(np.sum((~yt) & (~yp)))
    fn = int(np.sum(yt & (~yp)))
    n = tp + fp + tn + fn
    n_pos = tp + fn
    n_neg = fp + tn
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    tnr = tn / n_neg if n_neg > 0 else 0.0
    bal_acc = 0.5 * (rec + tnr)
    p_err = 1.0 - bal_acc
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    denom_mcc = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else 0.0
    auc = float(roc_auc_score(yt.astype(int), y_score)) if (y_score is not None and len(np.unique(yt)) == 2) else None
    return {
        "n_total": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy_pct": round(acc * 100.0, 2),
        "accuracy_wilson95_pct": list(MOD_03.wilson_ci(tp + tn, n)),
        "balanced_acc_pct": round(bal_acc * 100.0, 2),
        "p_err_pct": round(p_err * 100.0, 2),
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "recall_wilson95_pct": list(MOD_03.wilson_ci(tp, n_pos)),
        "f1_pct": round(f1 * 100.0, 2),
        "fpr_pct": round(fpr * 100.0, 2),
        "fpr_wilson95_pct": list(MOD_03.wilson_ci(fp, n_neg)),
        "mcc": round(mcc, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
    }


def method_payload_breakdown(records: List[Dict], y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute recall breakdown by stego method and payload plus dataset-level rates."""
    out: Dict[str, Any] = {
        "by_method_pct": {},
        "by_payload_pct": {},
        "clean_fpr_by_dataset_pct": {},
        "stego_recall_by_dataset_pct": {},
        "lsb_recall_by_dataset_pct": {},
    }
    for m in METHODS:
        idx = [i for i, r in enumerate(records) if r["cls"] == "Stego" and r["method"] == m]
        if idx:
            out["by_method_pct"][m] = round(float(np.mean(y_pred[idx])) * 100.0, 2)
    for pk in PAYLOADS:
        idx = [i for i, r in enumerate(records) if r["cls"] == "Stego" and r["payload"] == pk]
        if idx:
            out["by_payload_pct"][pk] = round(float(np.mean(y_pred[idx])) * 100.0, 2)
    for ds in DATASETS:
        idx_c = [i for i, r in enumerate(records) if r["cls"] == "Clean" and r["dataset"] == ds]
        idx_s = [i for i, r in enumerate(records) if r["cls"] == "Stego" and r["dataset"] == ds]
        idx_l = [i for i, r in enumerate(records) if r["cls"] == "Stego" and r["method"] == "LSB" and r["dataset"] == ds]
        if idx_c:
            out["clean_fpr_by_dataset_pct"][ds] = round(float(np.mean(y_pred[idx_c])) * 100.0, 2)
        if idx_s:
            out["stego_recall_by_dataset_pct"][ds] = round(float(np.mean(y_pred[idx_s])) * 100.0, 2)
        if idx_l:
            out["lsb_recall_by_dataset_pct"][ds] = round(float(np.mean(y_pred[idx_l])) * 100.0, 2)
    return out


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    """Compute continuity-corrected McNemar test and exact binomial p-value between two classifiers."""
    correct_a = pred_a.astype(bool) == y_true.astype(bool)
    correct_b = pred_b.astype(bool) == y_true.astype(bool)
    n_a_only = int(np.sum(correct_a & (~correct_b)))
    n_b_only = int(np.sum((~correct_a) & correct_b))
    tot_disc = n_a_only + n_b_only
    if tot_disc == 0:
        return {"n_a_right_b_wrong": 0, "n_b_right_a_wrong": 0, "chi2_cc": 0.0, "p_exact_binom": 1.0}
    chi2_cc = ((abs(n_a_only - n_b_only) - 1.0) ** 2) / float(tot_disc) if tot_disc > 0 else 0.0
    p_ex = float(binomtest(min(n_a_only, n_b_only), tot_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "n_a_right_b_wrong": n_a_only,
        "n_b_right_a_wrong": n_b_only,
        "chi2_cc": round(chi2_cc, 4),
        "p_exact_binom": p_ex,
    }


def run_exp1_ablation(m2_records: List[Dict]) -> Dict[str, Any]:
    print("=== Exp 1: Out-of-Fold 5-Fold CV Mode II Ablation Study (12,480 images) ===")
    y_true = np.array([1 if r["cls"] == "Stego" else 0 for r in m2_records], dtype=int)
    m1 = np.array([r["M1"] for r in m2_records], dtype=bool)
    m2 = np.array([r["M2"] for r in m2_records], dtype=bool)
    m3 = np.array([r["M3"] for r in m2_records], dtype=bool)
    s1 = np.array([r["s1"] for r in m2_records], dtype=np.float64)
    s2 = np.array([r["s2"] for r in m2_records], dtype=np.float64)
    s3 = np.array([r["s3"] for r in m2_records], dtype=np.float64)

    configs = [
        ("C1_Only", m1, s1),
        ("C2_Only", m2, s2),
        ("C3_Only", m3, s3),
        ("C1_plus_C2", m1 | m2, np.maximum(s1, s2)),
        ("C1_plus_C3", m1 | m3, np.maximum(s1, s3)),
        ("C2_plus_C3", m2 | m3, np.maximum(s2, s3)),
        ("Proposed_Mode_II_5FoldCV", m1 | m2 | m3, np.maximum.reduce([s1, s2, s3])),
    ]

    ablation_table = {}
    for name, pred, sc in configs:
        res = compute_binary_metrics(y_true, pred, sc)
        res.update(method_payload_breakdown(m2_records, pred))
        ablation_table[name] = res
        print(
            f"  {name:26s} | Acc={res['accuracy_pct']:5.2f}% | BalAcc={res['balanced_acc_pct']:5.2f}% "
            f"| Rec={res['recall_pct']:5.2f}% | FPR={res['fpr_pct']:4.2f}% {res['fpr_wilson95_pct']} | AUC={res['roc_auc']:.4f}"
        )

    pred_full = m1 | m2 | m3
    failure_cells = []
    for m in METHODS:
        for pk in PAYLOADS:
            for ds in DATASETS:
                idx = [
                    i
                    for i, r in enumerate(m2_records)
                    if r["cls"] == "Stego" and r["method"] == m and r["payload"] == pk and r["dataset"] == ds
                ]
                fn_cnt = int(np.sum(~pred_full[idx]))
                rec_cell = (len(idx) - fn_cnt) / len(idx) * 100.0
                if fn_cnt > 0:
                    failure_cells.append(
                        {"method": m, "payload": pk, "dataset": ds, "n": len(idx), "fn": fn_cnt, "recall_pct": round(rec_cell, 2)}
                    )
    failure_cells.sort(key=lambda x: x["recall_pct"])
    return {"ablation_full_12480": ablation_table, "worst_failure_cells": failure_cells[:12]}


def run_exp2_baselines(
    m2_records: List[Dict], base_records: List[Dict], X_spam: np.ndarray, manifest: Dict, out_dir: Path
) -> Tuple[Dict[str, Any], Dict]:
    print("\n=== Exp 2: Blind Baseline Comparison (Chi2, RS, SPA [3], SPAM-SVM 5-Fold CV vs Mode II 5-Fold CV) ===")
    assert len(m2_records) == len(base_records) == X_spam.shape[0]
    y_true = np.array([1 if r["cls"] == "Stego" else 0 for r in base_records], dtype=int)

    p_chi2_full = np.array([r["chi2_full_p"] for r in base_records], dtype=np.float64)
    p_chi2_pref = np.array([r["chi2_prefix10_p"] for r in base_records], dtype=np.float64)
    rs_full_mean = np.array([r["rs_full_mean"] for r in base_records], dtype=np.float64)
    spa_full_mean = np.array([r["spa_full_mean"] for r in base_records], dtype=np.float64)

    cover_to_fold = MOD_03.build_cover_folds(manifest, n_splits=5, seed=3537)
    row_folds = np.array([cover_to_fold[(r["dataset"], r["fname"])] for r in base_records], dtype=int)

    tr0_mask = row_folds != 0
    tr0_indices = np.where(tr0_mask)[0]
    inner_dev = row_folds[tr0_indices] == 1
    inner_tr = ~inner_dev
    sc_inner = StandardScaler().fit(X_spam[tr0_indices][inner_tr])
    best_c = 10.0
    best_bac = -1.0
    for c_cand in [1.0, 10.0, 50.0]:
        clf = SVC(kernel="rbf", C=c_cand, gamma="scale", class_weight="balanced", cache_size=1500)
        clf.fit(sc_inner.transform(X_spam[tr0_indices][inner_tr]), y_true[tr0_indices][inner_tr])
        preds = clf.predict(sc_inner.transform(X_spam[tr0_indices][inner_dev]))
        bac = 0.5 * (
            np.mean(preds[y_true[tr0_indices][inner_dev] == 1] == 1)
            + np.mean(preds[y_true[tr0_indices][inner_dev] == 0] == 0)
        )
        if bac > best_bac:
            best_bac = bac
            best_c = c_cand

    svm_oof_score = np.zeros(len(base_records), dtype=np.float64)
    svm_oof_pred = np.zeros(len(base_records), dtype=bool)
    fold_models = {}
    for fk in range(5):
        te_mask = row_folds == fk
        tr_mask = ~te_mask
        scaler = StandardScaler().fit(X_spam[tr_mask])
        clf = SVC(kernel="rbf", C=best_c, gamma="scale", class_weight="balanced", cache_size=1500)
        clf.fit(scaler.transform(X_spam[tr_mask]), y_true[tr_mask])
        te_scores = clf.decision_function(scaler.transform(X_spam[te_mask]))
        svm_oof_score[te_mask] = te_scores
        svm_oof_pred[te_mask] = te_scores >= 0.0
        fold_models[fk] = (scaler, clf)

    m2_by_key = {
        (r["cls"], r["payload"], r["method"], r["dataset"], r["fname"]): r
        for r in m2_records
    }
    m2_aligned = [
        m2_by_key[(r["cls"], r["payload"], r["method"], r["dataset"], r["fname"])]
        for r in base_records
    ]
    scores_m2 = np.array([r["score"] for r in m2_aligned], dtype=np.float64)
    pred_m2 = np.array([r["stego_pred"] for r in m2_aligned], dtype=bool)

    detectors = [
        ("Westfeld_Chi2_Full_p095", p_chi2_full >= 0.95, p_chi2_full),
        ("Westfeld_Chi2_Prefix10_p095", p_chi2_pref >= 0.95, p_chi2_pref),
        ("Fridrich_RS_Full_r005", rs_full_mean >= 0.05, rs_full_mean),
        ("Dumitrescu_SPA_Full_r005", spa_full_mean >= 0.05, spa_full_mean),
        ("Pevny_SPAM686_RBF_SVM_5FoldGroupCV", svm_oof_pred, svm_oof_score),
        ("Proposed_Mode_II_5FoldGroupCV", pred_m2, scores_m2),
    ]

    lsb_subset_idx = np.array([i for i, r in enumerate(base_records) if r["cls"] == "Clean" or r["method"] == "LSB"], dtype=int)
    lsb_records_sub = [base_records[i] for i in lsb_subset_idx]
    y_lsb = y_true[lsb_subset_idx]

    comparison: Dict[str, Any] = {"svm_tuned_c": best_c}
    lsb_only_comparison: Dict[str, Any] = {}
    roc_cache: Dict[str, Any] = {"y_true": y_true}

    for dname, pred, sc in detectors:
        res = compute_binary_metrics(y_true, pred, sc)
        res.update(method_payload_breakdown(base_records, pred))
        comparison[dname] = res

        sub_res = compute_binary_metrics(y_lsb, pred[lsb_subset_idx], sc[lsb_subset_idx])
        sub_res.update(method_payload_breakdown(lsb_records_sub, pred[lsb_subset_idx]))
        lsb_only_comparison[dname] = sub_res

        fpr_curve, tpr_curve, _ = roc_curve(y_true, sc)
        roc_cache[f"{dname}_fpr"] = fpr_curve.astype(np.float32)
        roc_cache[f"{dname}_tpr"] = tpr_curve.astype(np.float32)
        roc_cache[f"{dname}_auc"] = np.array([res["roc_auc"]], dtype=np.float32)

        print(
            f"  {dname:34s} | Acc={res['accuracy_pct']:5.2f}% | BalAcc={res['balanced_acc_pct']:5.2f}% "
            f"| Rec={res['recall_pct']:5.2f}% | FPR={res['fpr_pct']:5.2f}% | AUC={res['roc_auc']:.4f} "
            f"| LSB_BalAcc={sub_res['balanced_acc_pct']:5.2f}%"
        )

    mcnemar_all = mcnemar_test(y_true, pred_m2, svm_oof_pred)
    clean_mask = y_true == 0
    stego_mask = y_true == 1
    mcnemar_clean = mcnemar_test(y_true[clean_mask], pred_m2[clean_mask], svm_oof_pred[clean_mask])
    mcnemar_stego = mcnemar_test(y_true[stego_mask], pred_m2[stego_mask], svm_oof_pred[stego_mask])
    print(f"  McNemar Mode II vs SPAM-SVM -> All: {mcnemar_all} | Clean: {mcnemar_clean} | Stego: {mcnemar_stego}")

    comparison["lsb_only_subset_4800"] = lsb_only_comparison
    comparison["mcnemar_mode2_vs_spamsvm"] = {
        "all_12480": mcnemar_all,
        "clean_960": mcnemar_clean,
        "stego_11520": mcnemar_stego,
    }
    np.savez_compressed(out_dir / "ROC_Curves_Data.npz", **roc_cache)
    return comparison, {"cover_to_fold": cover_to_fold, "fold_models": fold_models, "row_folds": row_folds}


def _rate_worker(args: Tuple[int, str, str, Path, Dict[int, np.ndarray], Dict[int, Tuple[int, int, int]]]) -> List[Tuple]:
    img_idx, ds, fname, clean_dir, rate_bits, _ = args
    cov = MOD_02.load_rgb(clean_dir / ds / fname)
    h, w, c = cov.shape
    n_tot_lsb = h * w * c
    no_w, no_h = w // 3 - 1, h // 3 - 1
    hb, wb = (h // 8) * 8, (w // 8) * 8
    max_pvd_bits = max(0, no_w * no_h * 4 * 3 * 2)
    n_complex_blocks = 0
    if hb >= 8 and wb >= 8:
        blks = cov[:hb, :wb, :].reshape(hb // 8, 8, wb // 8, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 8, 8)
        planes = (blks[:, None, :, :] >> MOD_01.SHIFTS_DESC[None, :, None, None]) & 1
        hd = np.sum(planes[:, :, :, :-1] != planes[:, :, :, 1:], axis=(2, 3))
        vd = np.sum(planes[:, :, :-1, :] != planes[:, :, 1:, :], axis=(2, 3))
        n_complex_blocks = int(np.sum((hd + vd) >= 34))

    out_rows = []
    for rate_b, bits in rate_bits.items():
        if n_tot_lsb >= len(bits) + 16:
            stg_l = MOD_01.embed_lsb(cov, bits)
            out_rows.append(
                (
                    f"LSB_{rate_b}B",
                    ds,
                    fname,
                    MOD_02.extract_mode2_features_array(stg_l),
                    MOD_04.westfeld_chi2_pov(stg_l)[0],
                    MOD_04.fridrich_rs_rgb(stg_l)[0],
                    MOD_04.dumitrescu_spa_rgb(stg_l)[0],
                    MOD_04.extract_spam_686(stg_l, t_lim=3),
                )
            )
        if len(bits) + 88 <= (w // 3 - 1) * (h // 3 - 1) * 24:
            try:
                stg_p = MOD_01.embed_pvd_3x3(cov, bits)
                out_rows.append(
                    (
                        f"PVD_{rate_b}B",
                        ds,
                        fname,
                        MOD_02.extract_mode2_features_array(stg_p),
                        MOD_04.westfeld_chi2_pov(stg_p)[0],
                        MOD_04.fridrich_rs_rgb(stg_p)[0],
                        MOD_04.dumitrescu_spa_rgb(stg_p)[0],
                        MOD_04.extract_spam_686(stg_p, t_lim=3),
                    )
                )
            except AssertionError:
                pass
        req_blocks = int(math.ceil((len(bits) + 8) / 64.0))
        if n_complex_blocks >= req_blocks:
            stg_b = MOD_01.embed_bpcs_single_pass(cov, bits, alpha_t=0.30)
            out_rows.append(
                (
                    f"BPCS_{rate_b}B",
                    ds,
                    fname,
                    MOD_02.extract_mode2_features_array(stg_b),
                    MOD_04.westfeld_chi2_pov(stg_b)[0],
                    MOD_04.fridrich_rs_rgb(stg_b)[0],
                    MOD_04.dumitrescu_spa_rgb(stg_b)[0],
                    MOD_04.extract_spam_686(stg_b, t_lim=3),
                )
            )

    bits_770 = rate_bits[770] if 770 in rate_bits else np.unpackbits(
        np.frombuffer((MOD_01.LOREM_TEXT * 2).encode("utf-8")[:770], dtype=np.uint8)
    )
    stg_shuf = MOD_01.embed_lsb_shuffled(cov, bits_770, seed=3537 + img_idx)
    out_rows.append(
        (
            "LSB_Shuffled_770B",
            ds,
            fname,
            MOD_02.extract_mode2_features_array(stg_shuf),
            MOD_04.westfeld_chi2_pov(stg_shuf)[0],
            MOD_04.fridrich_rs_rgb(stg_shuf)[0],
            MOD_04.dumitrescu_spa_rgb(stg_shuf)[0],
            MOD_04.extract_spam_686(stg_shuf, t_lim=3),
        )
    )
    return out_rows


def run_exp3_variable_rates_and_shuffled(
    manifest: Dict, clean_dir: Path, m2_ctx: Dict, svm_ctx: Dict
) -> Dict[str, Any]:
    print("\n=== Exp 3: Variable Embedding Rates (256B, 512B, 2048B) & PRNG Key-Shuffled LSB (770B) ===")
    t0 = time.time()
    rate_bits: Dict[int, np.ndarray] = {}
    for rb in (256, 512, 770, 2048):
        raw_bytes = (MOD_01.LOREM_TEXT * 4).encode("utf-8")[:rb]
        rate_bits[rb] = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))

    tasks = [(i, r["dataset"], r["clean_fname"], clean_dir, {256: rate_bits[256], 512: rate_bits[512], 770: rate_bits[770], 2048: rate_bits[2048]}, {}) for i, r in enumerate(manifest["records"])]
    with ProcessPoolExecutor(max_workers=8) as ex:
        nested = list(ex.map(_rate_worker, tasks, chunksize=8))

    flat_rows = [item for sub in nested for item in sub]
    print(f"  Evaluated {len(flat_rows)} variable-rate & shuffled-LSB stego instances in {time.time() - t0:.2f}s")

    fold_th = m2_ctx["fold_thresholds"]
    cover_to_fold = svm_ctx["cover_to_fold"]
    fold_models = svm_ctx["fold_models"]

    by_cond: Dict[str, List[Tuple]] = {}
    for item in flat_rows:
        by_cond.setdefault(item[0], []).append(item)

    summary: Dict[str, Any] = {}
    for cond, grp in sorted(by_cond.items()):
        n = len(grp)
        m2_hits = 0
        svm_hits = 0
        chi2_hits = 0
        rs_hits = 0
        spa_hits = 0
        X_g = np.stack([g[7] for g in grp], axis=0)
        folds_g = np.array([cover_to_fold[(g[1], g[2])] for g in grp], dtype=int)
        svm_scores = np.zeros(n, dtype=np.float64)
        for fk in range(5):
            mask = folds_g == fk
            if np.any(mask):
                scaler, clf = fold_models[fk]
                svm_scores[mask] = clf.decision_function(scaler.transform(X_g[mask]))
        by_ds: Dict[str, List[int]] = {}
        for i, g in enumerate(grp):
            fk = int(folds_g[i])
            th = fold_th[str(fk)]
            _, _, _, pred_m2 = MOD_03.decide_mode2_vector(g[3], th)
            m2_hits += int(pred_m2)
            by_ds.setdefault(g[1], []).append(int(pred_m2))
            svm_hits += int(svm_scores[i] >= 0.0)
            chi2_hits += int(g[4] >= 0.95)
            rs_hits += int(g[5] >= 0.05)
            spa_hits += int(g[6] >= 0.05)

        summary[cond] = {
            "n_admissible": n,
            "mode2_5fold_recall_pct": round(m2_hits / n * 100.0, 2),
            "mode2_by_dataset_pct": {ds: round(sum(v) / len(v) * 100.0, 2) for ds, v in sorted(by_ds.items())},
            "spam_svm_5fold_recall_pct": round(svm_hits / n * 100.0, 2),
            "chi2_full_recall_pct": round(chi2_hits / n * 100.0, 2),
            "rs_full_recall_pct": round(rs_hits / n * 100.0, 2),
            "spa_full_recall_pct": round(spa_hits / n * 100.0, 2),
        }
        print(
            f"  {cond:20s} (n={n:3d}) | Mode II={summary[cond]['mode2_5fold_recall_pct']:6.2f}% "
            f"| SPAM-SVM={summary[cond]['spam_svm_5fold_recall_pct']:6.2f}% "
            f"| RS={summary[cond]['rs_full_recall_pct']:5.2f}% | SPA={summary[cond]['spa_full_recall_pct']:5.2f}%"
        )
    return summary


def _perturb_worker(args: Tuple[int, str, str, str, str, Path, Path]) -> List[Tuple[str, float, float, float, Dict[str, float], np.ndarray]]:
    img_idx, cls, method, ds, fname, clean_dir, stego_dir = args
    path = clean_dir / ds / fname if cls == "Clean" else stego_dir / "1_Plaintext_LoremIpsum" / method / ds / fname
    with Image.open(path) as im:
        base_pil = im.convert("RGB")
        base_arr = np.array(base_pil, dtype=np.uint8)

    results = []
    for r_idx, (rname, rtype, param) in enumerate(REGIMES):
        if rtype == "jpeg":
            buf = io.BytesIO()
            base_pil.save(buf, format="JPEG", quality=int(param))
            buf.seek(0)
            with Image.open(buf) as jim:
                p_arr = np.array(jim.convert("RGB"), dtype=np.uint8)
        elif rtype == "awgn":
            rng = np.random.default_rng(3537 + img_idx * 31 + r_idx)
            noise = rng.normal(0.0, float(param), size=base_arr.shape)
            p_arr = np.clip(np.rint(base_arr.astype(np.float64) + noise), 0, 255).astype(np.uint8)
        elif rtype == "crop":
            px = int(param)
            p_arr = base_arr[px:, px:, :].copy()
        else:
            nw = max(48, int(round(base_pil.width * float(param))))
            nh = max(48, int(round(base_pil.height * float(param))))
            p_arr = np.array(base_pil.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.uint8)

        p_chi2, _, _ = MOD_04.westfeld_chi2_pov(p_arr)
        rs_m, _ = MOD_04.fridrich_rs_rgb(p_arr)
        spa_m, _ = MOD_04.dumitrescu_spa_rgb(p_arr)
        f_m2 = MOD_02.extract_mode2_features_array(p_arr)
        spam_vec = MOD_04.extract_spam_686(p_arr, t_lim=3)
        results.append((rname, p_chi2, rs_m, spa_m, f_m2, spam_vec))
    return results


def run_exp4_robustness(
    manifest: Dict, clean_dir: Path, stego_dir: Path, m2_ctx: Dict, svm_ctx: Dict
) -> Dict[str, Any]:
    print("\n=== Exp 4: Post-Processing & Spatial Distortion Robustness on P1 (8 Regimes x 3,840 Images = 30,720 Eval) ===")
    t0 = time.time()
    tasks = []
    meta = []
    idx = 0
    for r in manifest["records"]:
        tasks.append((idx, "Clean", "Clean", r["dataset"], r["clean_fname"], clean_dir, stego_dir))
        meta.append({"cls": "Clean", "method": "Clean", "dataset": r["dataset"], "fname": r["clean_fname"], "payload": "Clean"})
        idx += 1
    for m in METHODS:
        for r in manifest["records"]:
            tasks.append((idx, "Stego", m, r["dataset"], r["clean_fname"], clean_dir, stego_dir))
            meta.append({"cls": "Stego", "method": m, "dataset": r["dataset"], "fname": r["clean_fname"], "payload": "1_Plaintext_LoremIpsum"})
            idx += 1

    with ProcessPoolExecutor(max_workers=8) as ex:
        rows_per_img = list(ex.map(_perturb_worker, tasks, chunksize=32))
    print(f"  Distortion extraction finished in {time.time() - t0:.2f}s")

    y_true = np.array([1 if r["cls"] == "Stego" else 0 for r in meta], dtype=int)
    folds = np.array([svm_ctx["cover_to_fold"][(r["dataset"], r["fname"])] for r in meta], dtype=int)
    fold_models = svm_ctx["fold_models"]
    fold_th = m2_ctx["fold_thresholds"]

    robust_summary: Dict[str, Any] = {}
    for r_idx, (rname, _, _) in enumerate(REGIMES):
        p_chi2 = np.array([row[r_idx][1] for row in rows_per_img], dtype=np.float64)
        rs_m = np.array([row[r_idx][2] for row in rows_per_img], dtype=np.float64)
        spa_m = np.array([row[r_idx][3] for row in rows_per_img], dtype=np.float64)
        m2_feats = [row[r_idx][4] for row in rows_per_img]
        X_pert = np.stack([row[r_idx][5] for row in rows_per_img], axis=0)

        m2_pred = np.zeros(len(meta), dtype=bool)
        m2_score = np.zeros(len(meta), dtype=np.float64)
        for i, feats in enumerate(m2_feats):
            th = fold_th[str(int(folds[i]))]
            _, _, _, pr = MOD_03.decide_mode2_vector(feats, th)
            s1, s2, s3 = MOD_03.mode2_component_scores(feats, th)
            m2_pred[i] = pr
            m2_score[i] = max(s1, s2, s3)

        svm_scores = np.zeros(len(meta), dtype=np.float64)
        svm_pred = np.zeros(len(meta), dtype=bool)
        for fk in range(5):
            mask = folds == fk
            scaler, clf = fold_models[fk]
            sc = clf.decision_function(scaler.transform(X_pert[mask]))
            svm_scores[mask] = sc
            svm_pred[mask] = sc >= 0.0

        regime_out: Dict[str, Any] = {}
        for dname, pred, score in [
            ("Proposed_Mode_II_CV", m2_pred, m2_score),
            ("SPAM_SVM_CleanTrained", svm_pred, svm_scores),
            ("Fridrich_RS_Full", rs_m >= 0.05, rs_m),
            ("Dumitrescu_SPA_Full", spa_m >= 0.05, spa_m),
            ("Westfeld_Chi2_Full", p_chi2 >= 0.95, p_chi2),
        ]:
            m_res = compute_binary_metrics(y_true, pred, score)
            m_res["by_method_pct"] = {
                m: round(float(np.mean(pred[[i for i, r in enumerate(meta) if r["cls"] == "Stego" and r["method"] == m]])) * 100.0, 2)
                for m in METHODS
            }
            regime_out[dname] = m_res
        robust_summary[rname] = regime_out
        m2r = regime_out["Proposed_Mode_II_CV"]
        svr = regime_out["SPAM_SVM_CleanTrained"]
        print(
            f"  {rname:12s} | Mode II BalAcc={m2r['balanced_acc_pct']:5.2f}% FPR={m2r['fpr_pct']:5.2f}% Rec={m2r['recall_pct']:5.2f}% "
            f"| SVM BalAcc={svr['balanced_acc_pct']:5.2f}% FPR={svr['fpr_pct']:5.2f}%"
        )
    return robust_summary


def gwet_ac1(labels_a: List[str], labels_b: List[str], cats: List[str]) -> float:
    """Compute exact Gwet's AC1 chance-corrected reliability coefficient."""
    n = len(labels_a)
    pa = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    q = len(cats)
    pe_parts = []
    for c in cats:
        pi_k = (sum(1 for a in labels_a if a == c) + sum(1 for b in labels_b if b == c)) / (2.0 * n)
        pe_parts.append(pi_k * (1.0 - pi_k))
    pe = sum(pe_parts) / max(1, q - 1)
    return float((pa - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 1.0


def run_exp5_consistency(manifest: Dict, clean_dir: Path, stego_dir: Path, arch_stg_dir: Path, out_dir: Path) -> Dict[str, Any]:
    print("\n=== Exp 5: Consistency Between Objective Eq. (9) Tiers and Salience / Archival Labels ===")
    eq9_labels = []
    svis_labels = []
    for m in METHODS:
        for r in manifest["records"]:
            cov = MOD_02.load_rgb(clean_dir / r["dataset"] / r["clean_fname"])
            stg = MOD_02.load_rgb(stego_dir / "1_Plaintext_LoremIpsum" / m / r["dataset"] / r["clean_fname"])
            res = MOD_02.compute_mode1_pair_array(cov, stg, method=m, compute_comp3=False)
            eq9_labels.append(res["vis_class"])
            sv = res["s_vis"]
            svis_labels.append("Clear" if sv >= 25.0 else ("Faint" if sv >= 15.0 else "Invisible"))

    cats = ["Clear", "Faint", "Invisible"]
    pa_u = sum(1 for a, b in zip(eq9_labels, svis_labels) if a == b) / len(eq9_labels)
    k_u_unw = float(cohen_kappa_score(eq9_labels, svis_labels, labels=cats))
    k_u_quad = float(cohen_kappa_score(eq9_labels, svis_labels, labels=cats, weights="quadratic"))
    ac1_u = gwet_ac1(eq9_labels, svis_labels, cats)

    thumb_cache = out_dir / "cover_thumb8x8_full_cache.npz"
    paths_cache = out_dir / "cover_thumb8x8_full_paths.json"
    npz = np.load(thumb_cache)
    with open(paths_cache, encoding="utf-8") as f:
        c_paths = json.load(f)

    manual_t35 = {
        ("LSB", "SSC_android_icon"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("LSB", "BCC_pokemon"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("LSB", "Gray"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("LSB", "FC_full_color"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("PVD", "SSC_android_icon"): {"Clear": 221, "Faint": 14, "Invisible": 5},
        ("PVD", "BCC_pokemon"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("PVD", "Gray"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("PVD", "FC_full_color"): {"Clear": 191, "Faint": 3, "Invisible": 1},
        ("BPCS", "SSC_android_icon"): {"Clear": 235, "Faint": 3, "Invisible": 2},
        ("BPCS", "BCC_pokemon"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("BPCS", "Gray"): {"Clear": 240, "Faint": 0, "Invisible": 0},
        ("BPCS", "FC_full_color"): {"Clear": 240, "Faint": 0, "Invisible": 0},
    }

    arch_eq9_counts: Dict[str, Dict[str, int]] = {}
    uniform_arch_man, uniform_arch_eq9 = [], []
    all_eq9_counts_agg: Counter[str] = Counter()
    all_man_counts_agg: Counter[str] = Counter()

    for ds in DATASETS:
        valid_idx = [i for i, p in enumerate(c_paths[ds]) if ("_gt" not in p.lower() and "\\Red Sea Bream GT\\" not in p and os.path.exists(p))]
        vecs = npz[ds][valid_idx]
        pths = [c_paths[ds][i] for i in valid_idx]
        for m in METHODS:
            folder = arch_stg_dir / m / ds
            fnames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
            c_eq9: Counter[str] = Counter()
            for fn in fnames:
                stg = MOD_02.load_rgb(folder / fn)
                h, w, c = stg.shape
                bh, bw = max(1, h // 8), max(1, w // 8)
                tv = stg[: bh * 8, : bw * 8, :].reshape(8, bh, 8, bw, c).mean(axis=(1, 3)).reshape(-1).astype(np.float32)
                dists = np.sum((vecs - tv) ** 2, axis=1)
                top5 = np.argsort(dists)[:5]
                cov = None
                best_mse = float("inf")
                for ti in top5:
                    cand = MOD_02.load_rgb(pths[int(ti)])
                    if cand.shape == stg.shape:
                        mse = float(np.mean((cand.astype(np.float64) - stg.astype(np.float64)) ** 2))
                        if mse < best_mse:
                            best_mse = mse
                            cov = cand
                assert cov is not None
                res = MOD_02.compute_mode1_pair_array(cov, stg, method=m, compute_comp3=False)
                lbl = res["vis_class"]
                c_eq9[lbl] += 1
                all_eq9_counts_agg[lbl] += 1
                if manual_t35[(m, ds)]["Faint"] == 0 and manual_t35[(m, ds)]["Invisible"] == 0:
                    uniform_arch_man.append("Clear")
                    uniform_arch_eq9.append(lbl)
            arch_eq9_counts[f"{m}/{ds}"] = {k: int(c_eq9[k]) for k in cats}
            for k in cats:
                all_man_counts_agg[k] += manual_t35[(m, ds)][k]

    min_a, min_b = [], []
    max_a, max_b = [], []
    for m in METHODS:
        for ds in DATASETS:
            man_d = manual_t35[(m, ds)]
            eq_d = arch_eq9_counts[f"{m}/{ds}"]
            if man_d["Faint"] == 0 and man_d["Invisible"] == 0:
                seq_m = ["Clear"] * man_d["Clear"]
                seq_e = ["Clear"] * eq_d["Clear"] + ["Faint"] * eq_d["Faint"] + ["Invisible"] * eq_d["Invisible"]
                min_a.extend(seq_m)
                min_b.extend(seq_e)
                max_a.extend(seq_m)
                max_b.extend(seq_e)
            else:
                rem_m = dict(man_d)
                rem_e = dict(eq_d)
                for c in cats:
                    match_k = min(rem_m[c], rem_e[c])
                    max_a.extend([c] * match_k)
                    max_b.extend([c] * match_k)
                    rem_m[c] -= match_k
                    rem_e[c] -= match_k
                max_a.extend([c for c in cats for _ in range(rem_m[c])])
                max_b.extend([c for c in cats for _ in range(rem_e[c])])
                min_a.extend([c for c in cats for _ in range(man_d[c])])
                min_b.extend([c for c in reversed(cats) for _ in range(eq_d[c])])

    pa_arch_min = sum(1 for a, b in zip(min_a, min_b) if a == b) / len(min_a)
    pa_arch_max = sum(1 for a, b in zip(max_a, max_b) if a == b) / len(max_a)
    k_arch_min = float(cohen_kappa_score(min_a, min_b, labels=cats))
    k_arch_max = float(cohen_kappa_score(max_a, max_b, labels=cats))
    ac1_arch_min = gwet_ac1(min_a, min_b, cats)
    ac1_arch_max = gwet_ac1(max_a, max_b, cats)

    pa_unif = sum(1 for a, b in zip(uniform_arch_man, uniform_arch_eq9) if a == b) / len(uniform_arch_man)
    ac1_unif = gwet_ac1(uniform_arch_man, uniform_arch_eq9, cats)

    print(f"  Unified P1 Eq.9 vs S_vis (n=2,880): PA={pa_u*100:.2f}%, kappa={k_u_unw:.4f} (quad={k_u_quad:.4f}), AC1={ac1_u:.4f}")
    print(f"  Archival Uniform-Clear Subset (n=2,160): PA={pa_unif*100:.2f}%, AC1={ac1_unif:.4f}")
    print(f"  Archival All 2,835 Bounds: PA=[{pa_arch_min*100:.2f}%, {pa_arch_max*100:.2f}%], AC1=[{ac1_arch_min:.4f}, {ac1_arch_max:.4f}]")

    return {
        "unified_p1_eq9_vs_svis_2880": {
            "n": len(eq9_labels),
            "percent_agreement_pct": round(pa_u * 100.0, 2),
            "cohen_kappa": round(k_u_unw, 4),
            "cohen_kappa_quadratic": round(k_u_quad, 4),
            "gwet_ac1": round(ac1_u, 4),
            "eq9_counts": dict(Counter(eq9_labels)),
            "svis_counts": dict(Counter(svis_labels)),
        },
        "archival_2835_vs_tables4_6": {
            "n_total": len(min_a),
            "manual_totals": dict(all_man_counts_agg),
            "eq9_totals": dict(all_eq9_counts_agg),
            "per_group_eq9": arch_eq9_counts,
            "uniform_clear_subset_n2160": {
                "n": len(uniform_arch_man),
                "percent_agreement_pct": round(pa_unif * 100.0, 2),
                "gwet_ac1": round(ac1_unif, 4),
            },
            "frechet_hoeffding_bounds": {
                "percent_agreement_pct": [round(pa_arch_min * 100.0, 2), round(pa_arch_max * 100.0, 2)],
                "cohen_kappa": [round(k_arch_min, 4), round(k_arch_max, 4)],
                "gwet_ac1": [round(ac1_arch_min, 4), round(ac1_arch_max, 4)],
            },
        },
    }


def run_exp6_runtime_ram(manifest: Dict, clean_dir: Path, stego_dir: Path, m2_ctx: Dict, svm_ctx: Dict) -> Dict[str, Any]:
    print("\n=== Exp 6: Single-Thread Runtime, Peak Heap Memory & Hardware Profile ===")
    tier_samples: Dict[str, List[Dict]] = {"SSC_64x64": [], "BCC_Gray_256x256": [], "FC_590x445": []}
    for r in manifest["records"]:
        if r["dataset"] == "SSC_android_icon" and r["width"] == 64 and r["height"] == 64 and len(tier_samples["SSC_64x64"]) < 50:
            tier_samples["SSC_64x64"].append(r)
        elif r["dataset"] in ("BCC_pokemon", "Gray") and len(tier_samples["BCC_Gray_256x256"]) < 50:
            tier_samples["BCC_Gray_256x256"].append(r)
        elif r["dataset"] == "FC_full_color" and len(tier_samples["FC_590x445"]) < 50:
            tier_samples["FC_590x445"].append(r)

    scaler0, clf0 = svm_ctx["fold_models"][0]
    th0 = m2_ctx["consensus_thresholds"]
    proc = psutil.Process(os.getpid())
    baseline_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    def bench_fn(pairs: List[Tuple[np.ndarray, np.ndarray]], mode: str) -> Dict[str, float]:
        times_ms = []
        tracemalloc.start()
        for cov, stg in pairs:
            t_s = time.perf_counter()
            if mode == "C1_Reveal_Bbox":
                cov_rev = (cov & 1) * 255
                rev = (stg & 1) * 255
                ys, xs, _ = np.where(cov != stg)
                if len(ys) > 0:
                    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                    _ = float(np.mean(np.abs(rev[y0:y1, x0:x1].astype(np.float32) - cov_rev[y0:y1, x0:x1].astype(np.float32))))
            elif mode == "Mode_I_Paired_Full":
                _ = MOD_02.compute_mode1_pair_array(cov, stg, method="LSB", compute_comp3=True)
            elif mode == "Mode_II_Blind":
                f = MOD_02.extract_mode2_features_array(stg)
                _ = MOD_03.decide_mode2_vector(f, th0)
            elif mode == "Westfeld_Chi2_Full":
                _ = MOD_04.westfeld_chi2_pov(stg)
            elif mode == "Fridrich_RS_Full":
                _ = MOD_04.fridrich_rs_rgb(stg)
            elif mode == "Dumitrescu_SPA_Full":
                _ = MOD_04.dumitrescu_spa_rgb(stg)
            elif mode == "Pevny_SPAM686_SVM":
                vec = MOD_04.extract_spam_686(stg, t_lim=3).reshape(1, -1)
                _ = clf0.decision_function(scaler0.transform(vec))
            times_ms.append((time.perf_counter() - t_s) * 1000.0)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "mean_ms": round(float(np.mean(times_ms)), 3),
            "std_ms": round(float(np.std(times_ms)), 3),
            "peak_alloc_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
        }

    pipelines = [
        "C1_Reveal_Bbox",
        "Mode_I_Paired_Full",
        "Mode_II_Blind",
        "Westfeld_Chi2_Full",
        "Fridrich_RS_Full",
        "Dumitrescu_SPA_Full",
        "Pevny_SPAM686_SVM",
    ]
    hw_info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "process_rss_mb": round(baseline_rss_mb, 2),
    }
    out: Dict[str, Any] = {"hardware": hw_info, "process_rss_mb": round(baseline_rss_mb, 2), "tiers": {}}
    for tname, recs in tier_samples.items():
        pairs = []
        for r in recs:
            cov = MOD_02.load_rgb(clean_dir / r["dataset"] / r["clean_fname"])
            stg = MOD_02.load_rgb(stego_dir / "1_Plaintext_LoremIpsum" / "LSB" / r["dataset"] / r["clean_fname"])
            pairs.append((cov, stg))
        h, w, _ = pairs[0][0].shape
        mpix = (h * w) / 1e6
        tier_res = {"n": len(pairs), "pixels": h * w, "megapixels": round(mpix, 4), "methods": {}}
        print(f"  Tier {tname} ({w}x{h} = {h * w:,} px, n={len(pairs)}):")
        for p_name in pipelines:
            bm = bench_fn(pairs, p_name)
            bm["ms_per_megapixel"] = round(bm["mean_ms"] / mpix, 2)
            tier_res["methods"][p_name] = bm
            print(
                f"    {p_name:22s}: {bm['mean_ms']:7.3f} +/- {bm['std_ms']:5.3f} ms | "
                f"peak_alloc={bm['peak_alloc_mb']:5.3f} MB | {bm['ms_per_megapixel']:7.2f} ms/MP"
            )
        out["tiers"][tname] = tier_res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute complete quantitative forensic benchmark suite.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    stego_dir = base_dir / "Real_Dataset" / "5_MultiPayload_Stegano"
    arch_stg_dir = base_dir / "Real_Dataset" / "3_Stegano_Sampled_240"
    out_dir = base_dir / "Code" / "output_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(out_dir / "Mode2_CV_Evaluations.json", encoding="utf-8") as f:
        m2_ctx = json.load(f)
    m2_records = m2_ctx["records"]
    with open(out_dir / "Classical_Baselines.json", encoding="utf-8") as f:
        base_ctx = json.load(f)
    base_records = base_ctx["records"]
    X_spam = np.load(out_dir / "SPAM686_Features.npz")["X"]

    out_path = out_dir / "Forensic_Benchmark_Results.json"
    summary: Dict[str, Any] = {
        "mode2_cv_spec_summary": m2_ctx["summary"],
        "mode2_cv_fold_thresholds": m2_ctx["fold_thresholds"],
        "mode2_cv_consensus_thresholds": m2_ctx["consensus_thresholds"],
        "dense_lsb_sanity_check": base_ctx.get("sanity_check_dense_lsb"),
    }

    summary["exp1_ablation_and_failures"] = run_exp1_ablation(m2_records)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    exp2, svm_ctx = run_exp2_baselines(m2_records, base_records, X_spam, manifest, out_dir)
    summary["exp2_baselines_comparison"] = exp2
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["exp3_variable_rates_and_shuffled"] = run_exp3_variable_rates_and_shuffled(manifest, clean_dir, m2_ctx, svm_ctx)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["exp5_eq9_salience_consistency"] = run_exp5_consistency(manifest, clean_dir, stego_dir, arch_stg_dir, out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["exp6_runtime_ram_complexity"] = run_exp6_runtime_ram(manifest, clean_dir, stego_dir, m2_ctx, svm_ctx)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["exp4_robustness_distortions"] = run_exp4_robustness(manifest, clean_dir, stego_dir, m2_ctx, svm_ctx)
    summary["total_elapsed_s"] = round(time.time() - t_all, 2)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== ALL FORENSIC BENCHMARKS COMPLETED in {time.time() - t_all:.2f}s ===")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
