import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.stats import chi2 as scipy_chi2

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = Path(__file__).resolve().parent.parent
PAYLOADS = [
    "1_Plaintext_LoremIpsum",
    "2_Structured_Binary",
    "3_Compressed_Zlib",
    "4_Encrypted_AES256",
]
METHODS = ["LSB", "PVD", "BPCS"]

AXIAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAG_DIRS = [(1, 1), (-1, -1), (1, -1), (-1, 1)]


def westfeld_chi2_pov(uint8_vals: np.ndarray) -> Tuple[float, float, int]:
    """Westfeld & Pfitzmann (2000) Chi-Square Pairs-of-Values statistical attack."""
    counts = np.bincount(uint8_vals.reshape(-1), minlength=256).astype(np.float64)
    n_even = counts[0::2]
    n_star = 0.5 * (counts[0::2] + counts[1::2])

    merged_even: List[float] = []
    merged_star: List[float] = []
    acc_e = 0.0
    acc_s = 0.0
    for ev, st in zip(n_even, n_star):
        if st <= 0.0:
            continue
        acc_e += float(ev)
        acc_s += float(st)
        if acc_s > 4.0:
            merged_even.append(acc_e)
            merged_star.append(acc_s)
            acc_e = 0.0
            acc_s = 0.0
    if acc_s > 0.0 and merged_star:
        merged_even[-1] += acc_e
        merged_star[-1] += acc_s

    k = len(merged_star)
    if k < 2:
        return 0.0, 0.0, 0
    me = np.asarray(merged_even, dtype=np.float64)
    ms = np.asarray(merged_star, dtype=np.float64)
    stat = float(np.sum(((me - ms) ** 2) / ms))
    df = k - 1
    p_val = float(scipy_chi2.sf(stat, df))
    return p_val, stat, df


def fridrich_rs_channel(plane_2d: np.ndarray) -> float:
    """Fridrich, Goljan, & Du (2001) RS steganalysis quadratic fill-rate estimator."""
    h, w = plane_2d.shape
    w4 = (w // 4) * 4
    if w4 < 4 or h < 1:
        return 0.0
    g = plane_2d[:, :w4].reshape(-1, 4).astype(np.int16)

    def disc(x: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(x[:, 1:] - x[:, :-1]), axis=1)

    def f_pos(x: np.ndarray) -> np.ndarray:
        return x ^ 1

    def f_neg(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        ev = (x & 1) == 0
        out[ev] = x[ev] - 1
        out[~ev] = x[~ev] + 1
        return np.clip(out, 0, 255)

    f0 = disc(g)
    gp = g.copy()
    gp[:, 1:3] = f_pos(gp[:, 1:3])
    gpp = disc(gp)
    gn = g.copy()
    gn[:, 1:3] = f_neg(gn[:, 1:3])
    gnn = disc(gn)

    n = float(len(f0))
    d0 = (float(np.sum(gpp > f0)) - float(np.sum(gpp < f0))) / n
    dn0 = (float(np.sum(gnn > f0)) - float(np.sum(gnn < f0))) / n

    gi = f_pos(g)
    fi0 = disc(gi)
    gip = gi.copy()
    gip[:, 1:3] = f_pos(gip[:, 1:3])
    fip = disc(gip)
    gin = gi.copy()
    gin[:, 1:3] = f_neg(gin[:, 1:3])
    fin = disc(gin)

    d1 = (float(np.sum(fip > fi0)) - float(np.sum(fip < fi0))) / n
    dn1 = (float(np.sum(fin > fi0)) - float(np.sum(fin < fi0))) / n

    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    dq = b * b - 4.0 * a * c
    if abs(a) > 1e-8 and dq >= 0.0:
        r1 = (-b + math.sqrt(dq)) / (2.0 * a)
        r2 = (-b - math.sqrt(dq)) / (2.0 * a)
        x = r1 if abs(r1) < abs(r2) else r2
        if abs(x - 0.5) > 1e-6:
            return float(max(0.0, min(1.5, x / (x - 0.5))))
    return 0.0


def fridrich_rs_rgb(img_rgb: np.ndarray) -> Tuple[float, float]:
    """Compute (mean_rgb_rs, max_rgb_rs) across the 3 RGB channels."""
    vals = [fridrich_rs_channel(img_rgb[:, :, ch]) for ch in range(3)]
    return float(np.mean(vals)), float(np.max(vals))


def dumitrescu_spa_channel(plane_2d: np.ndarray) -> float:
    """Dumitrescu, Wu, & Wang (2003) Sample Pair Analysis quadratic embedding-rate estimator."""
    uh = plane_2d[:, :-1].reshape(-1).astype(np.int32)
    vh = plane_2d[:, 1:].reshape(-1).astype(np.int32)
    uv = plane_2d[:-1, :].reshape(-1).astype(np.int32)
    vv = plane_2d[1:, :].reshape(-1).astype(np.int32)
    u = np.concatenate([uh, uv])
    v = np.concatenate([vh, vv])
    v_even = (v & 1) == 0
    xp = float(np.sum((v_even & (u < v)) | ((~v_even) & (u > v))))
    yp = float(np.sum((v_even & (u > v)) | ((~v_even) & (u < v))))
    wp = float(np.sum((u ^ 1) == v))
    zp = float(np.sum(u == v))
    p_tot = xp + yp + zp
    gamma = wp + zp
    a = 0.5 * gamma
    b = 2.0 * xp - p_tot
    c = yp - xp
    dq = b * b - 4.0 * a * c
    if a > 1e-8 and dq >= 0.0:
        r_minus = (-b - math.sqrt(dq)) / (2.0 * a)
        return float(max(0.0, min(1.0, r_minus)))
    return float(max(0.0, min(1.0, -b / (2.0 * a)))) if a > 1e-8 else 0.0


def dumitrescu_spa_rgb(img_rgb: np.ndarray) -> Tuple[float, float]:
    """Compute (mean_rgb_spa, max_rgb_spa) across R, G, B planes."""
    vals = [dumitrescu_spa_channel(img_rgb[:, :, ch]) for ch in range(3)]
    return float(np.mean(vals)), float(np.max(vals))


def extract_spam_686(img_rgb: np.ndarray, t_lim: int = 3) -> np.ndarray:
    """Pevny, Bas, & Fridrich (2010) 2nd-order SPAM feature vector (T=3 -> 686 dimensions)."""
    arr = img_rgb.astype(np.int16)
    h, w, _ = arr.shape
    base = 2 * t_lim + 1
    n_pairs = base * base
    n_states = n_pairs * base

    def direction_probs(dy: int, dx: int) -> np.ndarray:
        r0 = max(0, -dy)
        r1 = h - max(0, dy)
        c0 = max(0, -dx)
        c1 = w - max(0, dx)
        diff = arr[r0:r1, c0:c1, :] - arr[r0 + dy : r1 + dy, c0 + dx : c1 + dx, :]
        hd, wd, _ = diff.shape
        y0 = max(0, -2 * dy)
        y1 = hd - max(0, 2 * dy)
        x0 = max(0, -2 * dx)
        x1 = wd - max(0, 2 * dx)
        if y1 <= y0 or x1 <= x0:
            return np.zeros(n_states, dtype=np.float32)

        w_d = diff[y0:y1, x0:x1, :]
        v_d = diff[y0 + dy : y1 + dy, x0 + dx : x1 + dx, :]
        u_d = diff[y0 + 2 * dy : y1 + 2 * dy, x0 + 2 * dx : x1 + 2 * dx, :]
        mask = (
            (w_d >= -t_lim)
            & (w_d <= t_lim)
            & (v_d >= -t_lim)
            & (v_d <= t_lim)
            & (u_d >= -t_lim)
            & (u_d <= t_lim)
        )
        if not np.any(mask):
            return np.zeros(n_states, dtype=np.float32)

        idx = (
            (w_d[mask] + t_lim).astype(np.int32) * n_pairs
            + (v_d[mask] + t_lim).astype(np.int32) * base
            + (u_d[mask] + t_lim).astype(np.int32)
        )
        cnt = np.bincount(idx, minlength=n_states).astype(np.float64).reshape(n_pairs, base)
        row_sums = cnt.sum(axis=1, keepdims=True)
        probs = np.divide(cnt, row_sums, out=np.zeros_like(cnt), where=row_sums > 0.0)
        return probs.reshape(n_states).astype(np.float32)

    axial = np.mean([direction_probs(dy, dx) for dy, dx in AXIAL_DIRS], axis=0)
    diag = np.mean([direction_probs(dy, dx) for dy, dx in DIAG_DIRS], axis=0)
    return np.concatenate([axial, diag]).astype(np.float32)


def _process_one_full(args: Tuple[str, str, str, str, str, Path, Path]) -> Tuple[Dict, np.ndarray]:
    cls, pk, method, ds, fname, clean_dir, stego_dir = args
    path = clean_dir / ds / fname if cls == "Clean" else stego_dir / pk / method / ds / fname
    with Image.open(path) as im:
        img = np.array(im.convert("RGB"), dtype=np.uint8)
    h = img.shape[0]
    p_full, stat_full, _ = westfeld_chi2_pov(img)
    n_pref_rows = max(4, h // 10)
    p_pref10, _, _ = westfeld_chi2_pov(img[:n_pref_rows, :, :])
    rs_mean, rs_max = fridrich_rs_rgb(img)
    spa_mean, spa_max = dumitrescu_spa_rgb(img)
    spam_vec = extract_spam_686(img, t_lim=3)
    meta = {
        "cls": cls,
        "payload": pk,
        "method": method,
        "dataset": ds,
        "fname": fname,
        "chi2_full_p": round(p_full, 6),
        "chi2_full_stat": round(stat_full, 3),
        "chi2_prefix10_p": round(p_pref10, 6),
        "rs_full_mean": round(rs_mean, 6),
        "rs_full_max": round(rs_max, 6),
        "spa_full_mean": round(spa_mean, 6),
        "spa_full_max": round(spa_max, 6),
    }
    return meta, spam_vec


def _process_one_spa_only(args: Tuple[int, str, str, str, str, str, Path, Path]) -> Tuple[int, float, float]:
    idx, cls, pk, method, ds, fname, clean_dir, stego_dir = args
    path = clean_dir / ds / fname if cls == "Clean" else stego_dir / pk / method / ds / fname
    with Image.open(path) as im:
        img = np.array(im.convert("RGB"), dtype=np.uint8)
    spa_mean, spa_max = dumitrescu_spa_rgb(img)
    return idx, round(spa_mean, 6), round(spa_max, 6)


def run_dense_lsb_sanity_check(manifest: Dict, clean_dir: Path) -> Dict[str, Dict[str, float]]:
    """Verify Chi2, RS, and SPA implementations on sequential 40% and 100% LSB rates (Gray & FC covers)."""
    rng = np.random.default_rng(3537)
    out: Dict[str, Dict[str, float]] = {}
    for ds in ("Gray", "FC_full_color"):
        recs = [r for r in manifest["records"] if r["dataset"] == ds]
        ds_res: Dict[str, float] = {}
        for rate_label, rate in [("clean_0pct", 0.0), ("seq_lsb_40pct", 0.40), ("seq_lsb_100pct", 1.00)]:
            c2_f, c2_p, rs_f, sp_f = [], [], [], []
            for r in recs:
                with Image.open(clean_dir / ds / r["clean_fname"]) as im:
                    arr = np.array(im.convert("RGB"), dtype=np.uint8)
                if rate > 0.0:
                    flat = arr.reshape(-1)
                    k = int(rate * len(flat))
                    flat[:k] = (flat[:k] & 0xFE) | rng.integers(0, 2, size=k, dtype=np.uint8)
                pf, _, _ = westfeld_chi2_pov(arr)
                pp, _, _ = westfeld_chi2_pov(arr[: max(4, int(arr.shape[0] * max(0.1, min(1.0, rate)))), :, :])
                rm, _ = fridrich_rs_rgb(arr)
                sm, _ = dumitrescu_spa_rgb(arr)
                c2_f.append(pf >= 0.95)
                c2_p.append(pp >= 0.95)
                rs_f.append(rm >= 0.05)
                sp_f.append(sm >= 0.05)
            ds_res[rate_label] = {
                "n": len(recs),
                "chi2_full_pct": round(float(np.mean(c2_f)) * 100.0, 2),
                "chi2_matched_prefix_pct": round(float(np.mean(c2_p)) * 100.0, 2),
                "rs_ge005_pct": round(float(np.mean(rs_f)) * 100.0, 2),
                "spa_ge005_pct": round(float(np.mean(sp_f)) * 100.0, 2),
            }
        out[ds] = ds_res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Chi2, RS, SPA [3], and SPAM-686 baselines plus dense LSB sanity check.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    parser.add_argument("--merge-existing", action="store_true", help="Merge SPA into existing classical baseline cache instead of recomputing Chi2/RS/SPAM")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    stego_dir = base_dir / "Real_Dataset" / "5_MultiPayload_Stegano"
    out_dir = base_dir / "Code" / "output_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        manifest = json.load(f)

    sanity = run_dense_lsb_sanity_check(manifest, clean_dir)
    print(f"Dense sequential LSB sanity check completed in {time.time() - t0:.2f}s:")
    for ds, dvals in sanity.items():
        print(f"  {ds}: {dvals}")

    legacy_base = out_dir / "Langkah4_Classical_Baselines.json"
    legacy_spam = out_dir / "Langkah4_SPAM686_Features.npz"
    out_json = out_dir / "Classical_Baselines.json"
    out_npz = out_dir / "SPAM686_Features.npz"

    if args.merge_existing and legacy_base.exists() and legacy_spam.exists():
        print("Merging Dumitrescu SPA [3] into verified existing Chi2/RS/SPAM baseline records...")
        with open(legacy_base, encoding="utf-8") as f:
            existing = json.load(f)
        records = existing["records"]
        spa_tasks = [
            (i, r["cls"], r["payload"], r["method"], r["dataset"], r["fname"], clean_dir, stego_dir)
            for i, r in enumerate(records)
        ]
        with ProcessPoolExecutor(max_workers=8) as ex:
            spa_res = list(ex.map(_process_one_spa_only, spa_tasks, chunksize=64))
        for idx, sm, sx in spa_res:
            records[idx]["spa_full_mean"] = sm
            records[idx]["spa_full_max"] = sx
        if legacy_spam.resolve() != out_npz.resolve():
            X = np.load(legacy_spam)["X"]
            np.savez_compressed(out_npz, X=X)
    else:
        tasks = [("Clean", "Clean", "Clean", r["dataset"], r["clean_fname"], clean_dir, stego_dir) for r in manifest["records"]]
        for pk in PAYLOADS:
            for m in METHODS:
                for r in manifest["records"]:
                    tasks.append(("Stego", pk, m, r["dataset"], r["clean_fname"], clean_dir, stego_dir))
        print(f"Extracting Chi2, RS, SPA [3], and SPAM-686 across {len(tasks)} images...")
        with ProcessPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_process_one_full, tasks, chunksize=64))
        records = [r[0] for r in results]
        spam_matrix = np.stack([r[1] for r in results], axis=0).astype(np.float32)
        np.savez_compressed(out_npz, X=spam_matrix)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_images": len(records), "sanity_check_dense_lsb": sanity, "records": records}, f)

    print(f"Baselines & SPA extraction finished in {time.time() - t0:.2f}s | Saved: {out_json}")


if __name__ == "__main__":
    main()
