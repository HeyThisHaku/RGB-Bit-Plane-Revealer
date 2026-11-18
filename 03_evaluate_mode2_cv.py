import argparse
import importlib.util
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

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
FEATURE_KEYS = [
    "c1_lsb_trans_abs",
    "c1_pvd_h0_abs",
    "c1_pvd_b012_L",
    "c2_rs_top",
    "bpcs_col0_head",
    "uncorr_head",
    "rand40",
    "bpcs_inv_top",
]


def _load_extractor(code_dir: Path):
    spec = importlib.util.spec_from_file_location("m2_ext", str(code_dir / "02_extract_components_and_mode1.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute exact 95% Wilson score confidence interval (in percent) for binomial proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)


def build_cover_folds(manifest: Dict[str, Any], n_splits: int = 5, seed: int = 3537) -> Dict[Tuple[str, str], int]:
    """Assign each cover image deterministically to one stratified cross-validation fold."""
    rng = random.Random(seed)
    cover_to_fold: Dict[Tuple[str, str], int] = {}
    for ds in DATASETS:
        fnames = sorted(r["clean_fname"] for r in manifest["records"] if r["dataset"] == ds)
        rng.shuffle(fnames)
        for idx, fn in enumerate(fnames):
            cover_to_fold[(ds, fn)] = idx % n_splits
    return cover_to_fold


def calibrate_thresholds_from_split(tr_feats: np.ndarray, tr_y: np.ndarray) -> Dict[str, float]:
    """Calibrate all eight Mode II thresholds strictly inside a 4-fold training split."""
    clean_f = tr_feats[tr_y == 0]
    stego_f = tr_feats[tr_y == 1]

    def q_cands(col: int, q_list: List[float]) -> List[float]:
        return sorted(set(round(float(np.percentile(clean_f[:, col], q)), 4) for q in q_list))

    best_m1 = (0.20, 0.34, 0.06)
    best_m1_score = -1.0
    for t0 in q_cands(0, [98.8, 99.0, 99.2, 99.4, 99.6, 99.8]):
        m0_c = clean_f[:, 0] >= t0
        if float(np.mean(m0_c)) > 0.012:
            continue
        m0_s = stego_f[:, 0] >= t0
        for t1 in q_cands(1, [98.8, 99.0, 99.2, 99.4, 99.6, 99.8]):
            m01_c = m0_c | (clean_f[:, 1] >= t1)
            if float(np.mean(m01_c)) > 0.012:
                continue
            m01_s = m0_s | (stego_f[:, 1] >= t1)
            for t2 in q_cands(2, [98.8, 99.0, 99.2, 99.4, 99.6, 99.8]):
                fpr = float(np.mean(m01_c | (clean_f[:, 2] >= t2)))
                if fpr > 0.012:
                    continue
                rec = float(np.mean(m01_s | (stego_f[:, 2] >= t2)))
                sc = rec - 2.0 * fpr
                if sc > best_m1_score:
                    best_m1_score = sc
                    best_m1 = (t0, t1, t2)

    best_m2 = (0.18, 0.18)
    best_m2_score = -1.0
    for t3 in q_cands(3, [98.5, 99.0, 99.2, 99.4, 99.6, 99.8]):
        for t4 in q_cands(4, [98.5, 99.0, 99.2, 99.4, 99.6, 99.8]):
            fpr = float(np.mean((clean_f[:, 3] >= t3) | (clean_f[:, 4] >= t4)))
            if fpr > 0.012:
                continue
            rec = float(np.mean((stego_f[:, 3] >= t3) | (stego_f[:, 4] >= t4)))
            sc = rec - 2.0 * fpr
            if sc > best_m2_score:
                best_m2_score = sc
                best_m2 = (t3, t4)

    best_m3 = (0.17, 0.90, 0.05)
    best_m3_score = -1.0
    for t5 in q_cands(5, [90.0, 92.5, 95.0, 96.5, 97.5, 98.5]):
        for t6 in [0.80, 0.85, 0.90, 0.95]:
            sub_c = (clean_f[:, 5] >= t5) & (clean_f[:, 6] >= t6)
            if float(np.mean(sub_c)) > 0.020:
                continue
            sub_s = (stego_f[:, 5] >= t5) & (stego_f[:, 6] >= t6)
            for t7 in q_cands(7, [97.5, 98.0, 98.5, 99.0, 99.5]):
                fpr = float(np.mean(sub_c | (clean_f[:, 7] >= t7)))
                if fpr > 0.020:
                    continue
                rec = float(np.mean(sub_s | (stego_f[:, 7] >= t7)))
                sc = rec - 2.0 * fpr
                if sc > best_m3_score:
                    best_m3_score = sc
                    best_m3 = (t5, t6, t7)

    vals = best_m1 + best_m2 + best_m3
    return {k: round(float(v), 4) for k, v in zip(FEATURE_KEYS, vals)}


def decide_mode2_vector(feats: Dict[str, float], th: Dict[str, float]) -> Tuple[bool, bool, bool, bool]:
    """Evaluate Eq. (10) Mode II decision returning (M1, M2, M3, stego_decision)."""
    m1 = (
        (feats["c1_lsb_trans_abs"] >= th["c1_lsb_trans_abs"])
        or (feats["c1_pvd_h0_abs"] >= th["c1_pvd_h0_abs"])
        or (feats["c1_pvd_b012_L"] >= th["c1_pvd_b012_L"])
    )
    m2 = (feats["c2_rs_top"] >= th["c2_rs_top"]) or (feats["bpcs_col0_head"] >= th["bpcs_col0_head"])
    m3 = (
        (feats["uncorr_head"] >= th["uncorr_head"]) and (feats["rand40"] >= th["rand40"])
    ) or (feats["bpcs_inv_top"] >= th["bpcs_inv_top"])
    return m1, m2, m3, (m1 or m2 or m3)


def mode2_component_scores(feats: Dict[str, float], th: Dict[str, float]) -> Tuple[float, float, float]:
    """Compute normalized continuous ratio scores where >= 1.0 corresponds strictly to firing a rule."""
    s1 = max(
        feats["c1_lsb_trans_abs"] / max(th["c1_lsb_trans_abs"], 1e-9),
        feats["c1_pvd_h0_abs"] / max(th["c1_pvd_h0_abs"], 1e-9),
        feats["c1_pvd_b012_L"] / max(th["c1_pvd_b012_L"], 1e-9),
    )
    s2 = max(
        feats["c2_rs_top"] / max(th["c2_rs_top"], 1e-9),
        feats["bpcs_col0_head"] / max(th["bpcs_col0_head"], 1e-9),
    )
    s3 = max(
        min(
            feats["uncorr_head"] / max(th["uncorr_head"], 1e-9),
            feats["rand40"] / max(th["rand40"], 1e-9),
        ),
        feats["bpcs_inv_top"] / max(th["bpcs_inv_top"], 1e-9),
    )
    return float(s1), float(s2), float(s3)


def _extract_img_worker(args: Tuple[str, str, str, str, str, Path, Path, Any]) -> Dict[str, Any]:
    cls, pk, method, ds, fname, clean_dir, stego_dir, ext_fn = args
    from PIL import Image

    path = clean_dir / ds / fname if cls == "Clean" else stego_dir / pk / method / ds / fname
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    return {"cls": cls, "pk": pk, "method": method, "dataset": ds, "fname": fname, "blind": ext_fn(arr)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mode II via strict 5-fold cover-grouped cross-validation.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    parser.add_argument("--from-cache", action="store_true", help="Read Mode2_PerImage_Features.json instead of re-reading PNGs")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    code_dir = base_dir / "Code"
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    stego_dir = base_dir / "Real_Dataset" / "5_MultiPayload_Stegano"
    out_dir = code_dir / "output_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        manifest = json.load(f)

    cache_path = out_dir / "Mode2_PerImage_Features.json"
    rows: List[Dict[str, Any]] = []
    if args.from_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        for r in cached["clean"]:
            rows.append(
                {"cls": "Clean", "payload": "Clean", "method": "Clean", "dataset": r["dataset"], "fname": r["fname"], "features": r["blind"]}
            )
        for r in cached["stego"]:
            rows.append(
                {"cls": "Stego", "payload": r["pk"], "method": r["method"], "dataset": r["dataset"], "fname": r["fname"], "features": r["blind"]}
            )
    else:
        mod02 = _load_extractor(code_dir)
        tasks = [("Clean", "Clean", "Clean", r["dataset"], r["clean_fname"], clean_dir, stego_dir, mod02.extract_mode2_features_array) for r in manifest["records"]]
        for pk in PAYLOADS:
            for m in METHODS:
                for r in manifest["records"]:
                    tasks.append(("Stego", pk, m, r["dataset"], r["clean_fname"], clean_dir, stego_dir, mod02.extract_mode2_features_array))
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(_extract_img_worker, tasks, chunksize=64))
        for r in res:
            rows.append(
                {"cls": r["cls"], "payload": r["pk"], "method": r["method"], "dataset": r["dataset"], "fname": r["fname"], "features": r["blind"]}
            )

    cover_to_fold = build_cover_folds(manifest, n_splits=5, seed=3537)
    row_folds = np.array([cover_to_fold[(r["dataset"], r["fname"])] for r in rows], dtype=int)
    y_true = np.array([1 if r["cls"] == "Stego" else 0 for r in rows], dtype=int)
    feat_mat = np.array([[r["features"][k] for k in FEATURE_KEYS] for r in rows], dtype=np.float64)

    fold_thresholds: Dict[str, Dict[str, float]] = {}
    eval_records: List[Dict[str, Any]] = [None] * len(rows)

    for fk in range(5):
        te_idx = np.where(row_folds == fk)[0]
        tr_idx = np.where(row_folds != fk)[0]
        th = calibrate_thresholds_from_split(feat_mat[tr_idx], y_true[tr_idx])
        fold_thresholds[str(fk)] = th
        for idx in te_idx:
            rec = rows[idx]
            m1, m2, m3, pred = decide_mode2_vector(rec["features"], th)
            s1, s2, s3 = mode2_component_scores(rec["features"], th)
            eval_records[idx] = {
                "cls": rec["cls"],
                "payload": rec["payload"],
                "method": rec["method"],
                "dataset": rec["dataset"],
                "fname": rec["fname"],
                "fold": int(fk),
                "features": rec["features"],
                "M1": bool(m1),
                "M2": bool(m2),
                "M3": bool(m3),
                "stego_pred": bool(pred),
                "s1": round(s1, 6),
                "s2": round(s2, 6),
                "s3": round(s3, 6),
                "score": round(max(s1, s2, s3), 6),
            }

    consensus_th = {
        k: round(float(np.median([fold_thresholds[str(fk)][k] for fk in range(5)])), 4)
        for k in FEATURE_KEYS
    }
    fold_th_stats = {
        k: {
            "mean": round(float(np.mean([fold_thresholds[str(fk)][k] for fk in range(5)])), 4),
            "std": round(float(np.std([fold_thresholds[str(fk)][k] for fk in range(5)], ddof=1)), 4),
            "median": consensus_th[k],
        }
        for k in FEATURE_KEYS
    }

    clean_eval = [r for r in eval_records if r["cls"] == "Clean"]
    stego_eval = [r for r in eval_records if r["cls"] == "Stego"]
    fp = sum(r["stego_pred"] for r in clean_eval)
    tp = sum(r["stego_pred"] for r in stego_eval)
    tn = len(clean_eval) - fp
    fn = len(stego_eval) - tp

    clean_fpr = round(fp / len(clean_eval) * 100.0, 2)
    clean_fpr_ci = wilson_ci(fp, len(clean_eval))
    stego_rec = round(tp / len(stego_eval) * 100.0, 2)
    stego_rec_ci = wilson_ci(tp, len(stego_eval))
    acc = round((tp + tn) / len(eval_records) * 100.0, 2)
    bal_acc = round(0.5 * (tp / len(stego_eval) + tn / len(clean_eval)) * 100.0, 2)
    auc = round(float(roc_auc_score(y_true, [r["score"] for r in eval_records])), 4)

    ds_short = {"SSC_android_icon": "SSC", "BCC_pokemon": "BCC", "Gray": "Gray", "FC_full_color": "FC"}
    clean_by_ds = {
        ds_short[ds]: round(sum(r["stego_pred"] for r in clean_eval if r["dataset"] == ds) / sum(1 for r in clean_eval if r["dataset"] == ds) * 100.0, 2)
        for ds in DATASETS
    }
    clean_by_layer = {
        "C1": round(sum(r["M1"] for r in clean_eval) / len(clean_eval) * 100.0, 2),
        "C2": round(sum(r["M2"] for r in clean_eval) / len(clean_eval) * 100.0, 2),
        "C3": round(sum(r["M3"] for r in clean_eval) / len(clean_eval) * 100.0, 2),
    }

    pk_key_map = {
        "1_Plaintext_LoremIpsum": "P1_Plaintext",
        "2_Structured_Binary": "P2_StructuredBinary",
        "3_Compressed_Zlib": "P3_CompressedZlib",
        "4_Encrypted_AES256": "P4_EncryptedCTR",
    }
    stego_breakdown: Dict[str, Any] = {}
    for pk in PAYLOADS:
        sub = [r for r in stego_eval if r["payload"] == pk]
        entry: Dict[str, float] = {"overall": round(sum(r["stego_pred"] for r in sub) / len(sub) * 100.0, 2)}
        for m in METHODS:
            sub_m = [r for r in sub if r["method"] == m]
            entry[m] = round(sum(r["stego_pred"] for r in sub_m) / len(sub_m) * 100.0, 2)
        stego_breakdown[pk_key_map[pk]] = entry
    stego_breakdown["All_11520"] = stego_rec

    spec_out = {
        "description": "Strict 5-Fold Cover-Grouped Out-of-Fold CV calibration for Mode II (zero data leakage)",
        "fold_thresholds": fold_thresholds,
        "fold_threshold_summary": fold_th_stats,
        "consensus_thresholds": consensus_th,
        "verified_oof_performance": {
            "n_total": len(eval_records),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy_pct": acc,
            "balanced_acc_pct": bal_acc,
            "roc_auc": auc,
            "clean_960_fpr_pct": clean_fpr,
            "clean_960_fpr_wilson95_pct": list(clean_fpr_ci),
            "stego_recall_pct": stego_rec,
            "stego_recall_wilson95_pct": list(stego_rec_ci),
            "clean_fpr_by_dataset_pct": clean_by_ds,
            "clean_fpr_by_layer_pct": clean_by_layer,
            "stego_recall_by_payload_method_pct": stego_breakdown,
        },
    }

    spec_path = out_dir / "Mode2_CV_Spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec_out, f, indent=2)

    eval_path = out_dir / "Mode2_CV_Evaluations.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_clean": len(clean_eval),
                "n_stego": len(stego_eval),
                "fold_thresholds": fold_thresholds,
                "consensus_thresholds": consensus_th,
                "summary": spec_out["verified_oof_performance"],
                "records": eval_records,
            },
            f,
        )

    print(f"5-Fold Cover-Grouped CV Mode II completed in {time.time() - t0:.2f}s")
    print(f"OOF Acc={acc:.2f}% | BalAcc={bal_acc:.2f}% | AUC={auc:.4f} | FPR={clean_fpr:.2f}% {clean_fpr_ci} | Rec={stego_rec:.2f}% {stego_rec_ci}")
    print(f"Saved: {spec_path}")


if __name__ == "__main__":
    main()
