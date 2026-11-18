import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import stats as scipy_stats

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
SHIFTS_DESC = np.arange(7, -1, -1, dtype=np.uint8)


def load_rgb(path: Path | str) -> np.ndarray:
    """Load an image file as a canonical (H, W, 3) uint8 RGB NumPy array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def shannon_entropy_bits(bit_arr: np.ndarray) -> float:
    """Binary Shannon entropy H_0 of a 0/1 bit array."""
    if bit_arr.size == 0:
        return 0.0
    p1 = float(np.mean(bit_arr))
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    p0 = 1.0 - p1
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def shannon_entropy_bytes(uint8_arr: np.ndarray) -> float:
    """Global byte-level Shannon entropy H_global across 256 intensity levels."""
    counts = np.bincount(uint8_arr.reshape(-1), minlength=256)
    probs = counts[counts > 0] / uint8_arr.size
    return float(-np.sum(probs * np.log2(probs)))


def chi_square_hist_div(cov_uint8: np.ndarray, stg_uint8: np.ndarray) -> float:
    """Compute symmetric 256-bin histogram chi-square divergence."""
    hc = np.bincount(cov_uint8.reshape(-1), minlength=256).astype(np.float64)
    hs = np.bincount(stg_uint8.reshape(-1), minlength=256).astype(np.float64)
    denom = hc + hs
    mask = denom > 0
    return float(np.sum(((hs[mask] - hc[mask]) ** 2) / denom[mask]))


def pov_asymmetry(arr_uint8: np.ndarray) -> float:
    """Compute mean normalized Pair-of-Values |h(2k) - h(2k+1)| / (h(2k) + h(2k+1)) asymmetry."""
    h = np.bincount(arr_uint8.reshape(-1), minlength=256).astype(np.float64)
    even = h[0::2]
    odd = h[1::2]
    denom = even + odd
    mask = denom > 0
    return float(np.mean(np.abs(even[mask] - odd[mask]) / denom[mask])) if np.any(mask) else 0.0


def fridrich_rs_estimate(plane_2d: np.ndarray) -> float:
    """Fridrich RS quadratic steganalysis fill-rate estimator on a 2D channel strip."""
    h, w = plane_2d.shape
    w4 = (w // 4) * 4
    if w4 < 4 or h < 1:
        return 0.0
    g = plane_2d[:, :w4].reshape(-1, 4).astype(np.int16)

    def disc(x: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(x[:, 1:] - x[:, :-1]), axis=1)

    def flip_pos(x: np.ndarray) -> np.ndarray:
        return x ^ 1

    def flip_neg(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        ev = (x & 1) == 0
        out[ev] = x[ev] - 1
        out[~ev] = x[~ev] + 1
        return np.clip(out, 0, 255)

    f0 = disc(g)
    gp = g.copy()
    gp[:, 1:3] = flip_pos(gp[:, 1:3])
    gn = g.copy()
    gn[:, 1:3] = flip_neg(gn[:, 1:3])
    fp = disc(gp)
    fn = disc(gn)

    n = float(len(f0))
    d0 = (float(np.sum(fp > f0)) - float(np.sum(fp < f0))) / n
    dn0 = (float(np.sum(fn > f0)) - float(np.sum(fn < f0))) / n

    g_inv = flip_pos(g)
    fi0 = disc(g_inv)
    gip = g_inv.copy()
    gip[:, 1:3] = flip_pos(gip[:, 1:3])
    gin = g_inv.copy()
    gin[:, 1:3] = flip_neg(gin[:, 1:3])
    fip = disc(gip)
    fin = disc(gin)

    d1 = (float(np.sum(fip > fi0)) - float(np.sum(fip < fi0))) / n
    dn1 = (float(np.sum(fin > fi0)) - float(np.sum(fin < fi0))) / n

    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    disc_q = b * b - 4.0 * a * c
    if abs(a) > 1e-8 and disc_q >= 0.0:
        r1 = (-b + math.sqrt(disc_q)) / (2.0 * a)
        r2 = (-b - math.sqrt(disc_q)) / (2.0 * a)
        x = r1 if abs(r1) < abs(r2) else r2
        if abs(x - 0.5) > 1e-6:
            return float(max(0.0, min(1.5, x / (x - 0.5))))
    return 0.0


def classify_eq9(delta_c_bbox: float, delta_h0_roi: float) -> str:
    """Partition active contrast and ROI entropy shifts into Eq. (9) visibility tiers."""
    if delta_c_bbox >= 45.0 or (delta_c_bbox >= 25.0 and delta_h0_roi >= 0.15):
        return "Clear"
    if delta_c_bbox >= 20.0:
        return "Faint"
    return "Invisible"


def extract_mode2_features_array(img_rgb: np.ndarray) -> Dict[str, float]:
    """Extract the eight payload-agnostic Mode II features without cover reference."""
    h, w, c = img_rgb.shape
    lsb_flat = (img_rgb & 1).reshape(-1)
    n_tot = len(lsb_flat)

    scales = [length for length in (2048, 4096, 8192) if length <= n_tot // 2]
    if not scales:
        scales = [max(512, n_tot // 4)]
    c1_lsb_trans_abs = max(
        abs(float(np.mean(lsb_flat[: length - 1] != lsb_flat[1:length])) - float(np.mean(lsb_flat[-length:-1] != lsb_flat[-length + 1 :])))
        for length in scales
    )

    k_rows = max(4, min(h // 4, int(math.ceil(2048.0 / w))))
    c2_rs_top = fridrich_rs_estimate(img_rgb[:k_rows, :, 0])

    nh = h // 3 - 1
    k_blks = min(nh, 64)
    bx0 = img_rgb[: k_blks * 3, :3, :].reshape(k_blks, 3, 3, 3)
    c_corners = bx0[:, [0, 0, 2, 2], [0, 2, 0, 2], :]
    s_cross = bx0[:, [0, 1, 1, 2], [1, 0, 2, 1], :]
    c1_pvd_h0_abs = abs(shannon_entropy_bits(c_corners.reshape(-1) & 1) - shannon_entropy_bits(s_cross.reshape(-1) & 1))

    blks0 = img_rgb[: k_blks * 3, :3, 0].reshape(k_blks, 3, 3)
    tot_ex = 0.0
    for sh in (0, 1, 2):
        bp = (blks0 >> sh) & 1
        c00, c02, c20, c22 = bp[:, 0, 0], bp[:, 0, 2], bp[:, 2, 0], bp[:, 2, 2]
        s01, s10, s12, s21, m11 = bp[:, 0, 1], bp[:, 1, 0], bp[:, 1, 2], bp[:, 2, 1], bp[:, 1, 1]
        c_tr = float(
            np.mean(
                np.concatenate(
                    [c00 != s01, c00 != s10, c02 != s01, c02 != s12, c20 != s10, c20 != s21, c22 != s12, c22 != s21]
                )
            )
        )
        s_tr = float(np.mean(np.concatenate([s01 != m11, s10 != m11, s12 != m11, s21 != m11])))
        tot_ex += c_tr - s_tr
    c1_pvd_b012_l = tot_ex / 3.0

    hb, wb = (h // 8) * 8, (w // 8) * 8
    blks8 = (
        img_rgb[:hb, :wb, :]
        .reshape(hb // 8, 8, wb // 8, 8, c)
        .transpose(0, 2, 4, 1, 3)
        .reshape(-1, 8, 8)
    )
    planes = (blks8[:, None, :, :] >> SHIFTS_DESC[None, :, None, None]) & 1
    hd = np.sum(planes[:, :, :, :-1] != planes[:, :, :, 1:], axis=(2, 3))
    vd = np.sum(planes[:, :, :-1, :] != planes[:, :, 1:, :], axis=(2, 3))
    alphas = (hd + vd) / 112.0

    bpcs_col0_head = 0.0
    eh = 0.5
    rand40 = 0.0
    idx_b, idx_p = np.where(alphas >= 0.30)
    if len(idx_b) >= 20:
        cp = planes[idx_b, idx_p]
        nc = len(cp)
        head = cp[: min(97, max(16, nc // 2))]
        bpcs_col0_head = float(abs(np.mean(head[:, :, 1:]) - np.mean(head[:, :, 0])))
        hs = np.abs(np.mean(head[:, :, :-1] == head[:, :, 1:], axis=(1, 2)) - 0.5)
        vs = np.abs(np.mean(head[:, :-1, :] == head[:, 1:, :], axis=(1, 2)) - 0.5)
        eh = float(np.mean(hs + vs))
        ah = (
            np.sum(head[:, :, :-1] != head[:, :, 1:], axis=(1, 2))
            + np.sum(head[:, :-1, :] != head[:, 1:, :], axis=(1, 2))
        ) / 112.0
        rand40 = float(np.mean((ah >= 0.40) & (ah <= 0.60)))
    uncorr_head = max(0.0, 0.30 - eh)

    has_cx = np.any(alphas >= 0.30, axis=1)
    cx_alphas = alphas[has_cx]
    bpcs_inv_top = 0.0
    if len(cx_alphas) >= 10:
        n_hd = min(35, max(8, len(cx_alphas) // 2))
        hd_a = cx_alphas[:n_hd]
        tl_a = cx_alphas[n_hd:] if len(cx_alphas) > 45 else cx_alphas[len(cx_alphas) // 2 :]
        inv_hd = float(np.mean(np.max(hd_a[:, :-1] - hd_a[:, 1:], axis=1)))
        inv_tl = float(np.mean(np.max(tl_a[:, :-1] - tl_a[:, 1:], axis=1))) if len(tl_a) > 0 else inv_hd
        bpcs_inv_top = inv_hd - inv_tl

    return {
        "c1_lsb_trans_abs": float(c1_lsb_trans_abs),
        "c1_pvd_h0_abs": float(c1_pvd_h0_abs),
        "c1_pvd_b012_L": float(c1_pvd_b012_l),
        "c2_rs_top": float(c2_rs_top),
        "bpcs_col0_head": float(bpcs_col0_head),
        "uncorr_head": float(uncorr_head),
        "rand40": float(rand40),
        "bpcs_inv_top": float(bpcs_inv_top),
    }


def compute_mode1_pair_array(
    cov: np.ndarray, stg: np.ndarray, method: str = "LSB", compute_comp3: bool = True
) -> Dict[str, Any]:
    """Compute paired Mode I Component 1, 2, and 3 metrics from cover and stego RGB arrays."""
    cov_rev = ((cov & 1) * 255).astype(np.float32)
    rev_comp = ((stg & 1) * 255).astype(np.uint8)
    rev_f = rev_comp.astype(np.float32)

    diff_mask = cov != stg
    mod_ch = int(np.sum(diff_mask))
    lsb_flips = int(np.sum((cov & 1) != (stg & 1)))
    ys, xs, _ = np.where(diff_mask)

    if method == "LSB":
        active_indices = np.flatnonzero(diff_mask.reshape(-1))
        end_pos = int(active_indices.max()) + 1 if active_indices.size > 0 else min(len(cov.reshape(-1)), 6176)
        roi_cov_bits = cov.reshape(-1)[:end_pos] & 1
        roi_stg_bits = stg.reshape(-1)[:end_pos] & 1
    else:
        if len(ys) > 0:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            roi_cov_bits = (cov[y0:y1, x0:x1, :] & 1).reshape(-1)
            roi_stg_bits = (stg[y0:y1, x0:x1, :] & 1).reshape(-1)
        else:
            roi_cov_bits = (cov & 1).reshape(-1)
            roi_stg_bits = (stg & 1).reshape(-1)

    if len(ys) > 0:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        delta_c_bbox = float(np.mean(np.abs(rev_f[y0:y1, x0:x1] - cov_rev[y0:y1, x0:x1])))
        cov_lsb_var = float(np.var(cov_rev[y0:y1, x0:x1]))
    else:
        delta_c_bbox = 0.0
        cov_lsb_var = float(np.var(cov_rev))

    h0_roi_cov = shannon_entropy_bits(roi_cov_bits)
    h0_roi_stg = shannon_entropy_bits(roi_stg_bits)
    delta_h0 = h0_roi_stg - h0_roi_cov
    s_vis = delta_c_bbox * (1.0 - 0.35 * min(1.0, cov_lsb_var / 16256.25))
    vis_class = classify_eq9(delta_c_bbox, delta_h0)

    h_glob_cov = shannon_entropy_bytes(cov)
    h_glob_stg = shannon_entropy_bytes(stg)

    cov_f = cov.astype(np.float64)
    stg_f = stg.astype(np.float64)
    mse = float(np.mean((cov_f - stg_f) ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 99.0

    mu_shift = abs(float(np.mean(stg_f)) - float(np.mean(cov_f)))
    var_shift = abs(float(np.var(stg_f)) - float(np.var(cov_f)))
    chi2 = chi_square_hist_div(cov, stg) if compute_comp3 else 0.0
    pov_cov = pov_asymmetry(cov) if compute_comp3 else 0.0
    pov_stg = pov_asymmetry(stg) if compute_comp3 else 0.0

    return {
        "rev_comp": rev_comp,
        "mod_ch": mod_ch,
        "lsb_flips": lsb_flips,
        "mse": mse,
        "psnr": psnr,
        "h_glob_cov": h_glob_cov,
        "h_glob_stg": h_glob_stg,
        "h0_roi_cov": h0_roi_cov,
        "h0_roi_stg": h0_roi_stg,
        "delta_h0": delta_h0,
        "delta_c_bbox": delta_c_bbox,
        "s_vis": s_vis,
        "vis_class": vis_class,
        "mu_shift": mu_shift,
        "var_shift": var_shift,
        "chi2": chi2,
        "pov_cov": pov_cov,
        "pov_stg": pov_stg,
    }


def _process_cover_bundle_worker(args: Tuple[str, str, Path, Path, Path, bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ds, fname, clean_dir, stego_dir, reveal_dir, verify_reveal = args
    cov = load_rgb(clean_dir / ds / fname)
    cov_blind = {"dataset": ds, "fname": fname, "blind": extract_mode2_features_array(cov)}
    cov_rev = ((cov & 1) * 255).astype(np.float32)
    cov_f = cov.astype(np.float64)
    h_glob_cov = shannon_entropy_bytes(cov)
    pov_cov = pov_asymmetry(cov)
    cov_mean = float(np.mean(cov_f))
    cov_var = float(np.var(cov_f))
    cov_flat_lsb = cov.reshape(-1) & 1

    stego_items: List[Dict[str, Any]] = []
    for pk in PAYLOADS:
        for method in METHODS:
            stg = load_rgb(stego_dir / pk / method / ds / fname)
            if verify_reveal:
                rev_disk = load_rgb(reveal_dir / pk / method / ds / fname)
                assert np.array_equal(rev_disk, ((stg & 1) * 255).astype(np.uint8))

            rev_f = ((stg & 1) * 255).astype(np.float32)
            diff_mask = cov != stg
            mod_ch = int(np.sum(diff_mask))
            lsb_flips = int(np.sum(cov_flat_lsb != (stg.reshape(-1) & 1)))
            ys, xs, _ = np.where(diff_mask)

            if method == "LSB":
                active_idx = np.flatnonzero(diff_mask.reshape(-1))
                end_pos = int(active_idx.max()) + 1 if active_idx.size > 0 else min(len(cov_flat_lsb), 6176)
                roi_cov_bits = cov_flat_lsb[:end_pos]
                roi_stg_bits = stg.reshape(-1)[:end_pos] & 1
            else:
                if len(ys) > 0:
                    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                    roi_cov_bits = (cov[y0:y1, x0:x1, :] & 1).reshape(-1)
                    roi_stg_bits = (stg[y0:y1, x0:x1, :] & 1).reshape(-1)
                else:
                    roi_cov_bits = cov_flat_lsb
                    roi_stg_bits = (stg & 1).reshape(-1)

            if len(ys) > 0:
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                delta_c_bbox = float(np.mean(np.abs(rev_f[y0:y1, x0:x1] - cov_rev[y0:y1, x0:x1])))
                cov_lsb_var = float(np.var(cov_rev[y0:y1, x0:x1]))
            else:
                delta_c_bbox = 0.0
                cov_lsb_var = float(np.var(cov_rev))

            h0_roi_cov = shannon_entropy_bits(roi_cov_bits)
            h0_roi_stg = shannon_entropy_bits(roi_stg_bits)
            delta_h0 = h0_roi_stg - h0_roi_cov
            s_vis = delta_c_bbox * (1.0 - 0.35 * min(1.0, cov_lsb_var / 16256.25))
            vis_class = classify_eq9(delta_c_bbox, delta_h0)

            h_glob_stg = shannon_entropy_bytes(stg)
            stg_f = stg.astype(np.float64)
            mse = float(np.mean((cov_f - stg_f) ** 2))
            psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 0 else 99.0

            is_p1 = pk == "1_Plaintext_LoremIpsum"
            mu_shift = abs(float(np.mean(stg_f)) - cov_mean) if is_p1 else 0.0
            var_shift = abs(float(np.var(stg_f)) - cov_var) if is_p1 else 0.0
            chi2 = chi_square_hist_div(cov, stg) if is_p1 else 0.0
            pov_stg = pov_asymmetry(stg) if is_p1 else 0.0

            stego_items.append(
                {
                    "pk": pk,
                    "method": method,
                    "dataset": ds,
                    "fname": fname,
                    "mod_ch": mod_ch,
                    "lsb_flips": lsb_flips,
                    "mse": mse,
                    "psnr": psnr,
                    "h_glob_cov": h_glob_cov,
                    "h_glob_stg": h_glob_stg,
                    "h0_roi_cov": h0_roi_cov,
                    "h0_roi_stg": h0_roi_stg,
                    "delta_c_bbox": delta_c_bbox,
                    "s_vis": s_vis,
                    "vis_class": vis_class,
                    "mu_shift": mu_shift,
                    "var_shift": var_shift,
                    "chi2": chi2,
                    "pov_cov": pov_cov if is_p1 else 0.0,
                    "pov_stg": pov_stg,
                    "blind": extract_mode2_features_array(stg),
                }
            )
    return cov_blind, stego_items


def paired_stats(x_cov: List[float], x_stg: List[float]) -> Tuple[float, float, float, float]:
    """Compute paired t-statistic, p_t, Wilcoxon signed-rank p_W, and paired Cohen's d."""
    diff = np.array(x_stg, dtype=np.float64) - np.array(x_cov, dtype=np.float64)
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohen_d = (mean_d / std_d) if std_d > 1e-12 else 0.0
    if std_d > 1e-12:
        t_stat, p_t = scipy_stats.ttest_rel(x_stg, x_cov)
        try:
            _, p_w = scipy_stats.wilcoxon(diff, zero_method="wilcox")
        except Exception:
            p_w = float(p_t)
        return float(t_stat), float(p_t), float(p_w), float(cohen_d)
    return 0.0, 1.0, 1.0, 0.0


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply step-down Holm-Bonferroni family-wise error correction."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paired Mode I metrics and payload-agnostic blind Mode II features.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    parser.add_argument("--verify-reveal", action="store_true", help="Cross-check reveal PNGs on disk byte-for-byte")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    real_dir = base_dir / "Real_Dataset"
    clean_dir = real_dir / "7_Unified_Clean_Covers_960"
    stego_dir = real_dir / "5_MultiPayload_Stegano"
    reveal_dir = real_dir / "6_MultiPayload_Revealed"
    out_dir = base_dir / "Code" / "output_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(clean_dir / "manifest_960.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cover_tasks = [
        (rec["dataset"], rec["clean_fname"], clean_dir, stego_dir, reveal_dir, args.verify_reveal)
        for rec in manifest["records"]
    ]
    print(f"Extracting Mode I paired & Mode II blind features across {len(cover_tasks)} cover bundles (12,480 images)...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        bundled = list(ex.map(_process_cover_bundle_worker, cover_tasks, chunksize=8))

    clean_res = [b[0] for b in bundled]
    stego_res = [item for b in bundled for item in b[1]]
    print(f"Extraction finished in {time.time() - t0:.2f}s ({len(clean_res)} clean + {len(stego_res)} stego)")

    by_pk_m_ds = defaultdict(list)
    for r in stego_res:
        by_pk_m_ds[(r["pk"], r["method"], r["dataset"])].append(r)

    multipayload_summary: Dict[str, Any] = {}
    for pk in PAYLOADS:
        multipayload_summary[pk] = {}
        for m in METHODS:
            multipayload_summary[pk][m] = {}
            all_rows = []
            for ds in DATASETS:
                rows = by_pk_m_ds[(pk, m, ds)]
                all_rows.extend(rows)
                vc = {"Clear": 0, "Faint": 0, "Invisible": 0}
                for r in rows:
                    vc[r["vis_class"]] += 1
                multipayload_summary[pk][m][ds] = {
                    "n": len(rows),
                    "mean_mod_ch": round(float(np.mean([r["mod_ch"] for r in rows])), 2),
                    "mean_lsb_flips": round(float(np.mean([r["lsb_flips"] for r in rows])), 2),
                    "lsb_flip_rate_pct": round(float(np.mean([r["lsb_flips"] for r in rows])) / 6160.0 * 100.0, 2),
                    "mean_mse": round(float(np.mean([r["mse"] for r in rows])), 4),
                    "mean_psnr_db": round(float(np.mean([r["psnr"] for r in rows])), 2),
                    "mean_h0_roi": round(float(np.mean([r["h0_roi_stg"] for r in rows])), 4),
                    "mean_h_global": round(float(np.mean([r["h_glob_stg"] for r in rows])), 4),
                    "mean_delta_c_bbox": round(float(np.mean([r["delta_c_bbox"] for r in rows])), 2),
                    "mean_s_vis": round(float(np.mean([r["s_vis"] for r in rows])), 2),
                    "eq9_counts": vc,
                }
            vc_all = {"Clear": 0, "Faint": 0, "Invisible": 0}
            for r in all_rows:
                vc_all[r["vis_class"]] += 1
            multipayload_summary[pk][m]["_ALL_960"] = {
                "n": len(all_rows),
                "mean_mod_ch": round(float(np.mean([r["mod_ch"] for r in all_rows])), 2),
                "mean_lsb_flips": round(float(np.mean([r["lsb_flips"] for r in all_rows])), 2),
                "lsb_flip_rate_pct": round(float(np.mean([r["lsb_flips"] for r in all_rows])) / 6160.0 * 100.0, 2),
                "mean_psnr_db": round(float(np.mean([r["psnr"] for r in all_rows])), 2),
                "mean_h0_roi": round(float(np.mean([r["h0_roi_stg"] for r in all_rows])), 4),
                "eq9_counts": vc_all,
            }

    entropy_summary: Dict[str, Any] = {}
    comp3_summary: Dict[str, Any] = {}
    pk0 = "1_Plaintext_LoremIpsum"
    raw_cells = []
    for m in METHODS:
        entropy_summary[m] = {}
        comp3_summary[m] = {}
        for ds in DATASETS:
            rows = by_pk_m_ds[(pk0, m, ds)]
            h_gc = [r["h_glob_cov"] for r in rows]
            h_gs = [r["h_glob_stg"] for r in rows]
            h0_rc = [r["h0_roi_cov"] for r in rows]
            h0_rs = [r["h0_roi_stg"] for r in rows]
            _, ptg, pwg, dg = paired_stats(h_gc, h_gs)
            _, ptr, pwr, dr = paired_stats(h0_rc, h0_rs)
            entropy_summary[m][ds] = {
                "n": len(rows),
                "H_global_cover": round(float(np.mean(h_gc)), 4),
                "H_global_stego": round(float(np.mean(h_gs)), 4),
                "Delta_H_global": round(float(np.mean(h_gs) - np.mean(h_gc)), 4),
                "p_glob_ttest": ptg,
                "p_glob_wilcoxon": pwg,
                "cohen_d_glob": round(dg, 3),
                "H0_ROI_cover": round(float(np.mean(h0_rc)), 4),
                "H0_ROI_stego": round(float(np.mean(h0_rs)), 4),
                "Delta_H0_ROI": round(float(np.mean(h0_rs) - np.mean(h0_rc)), 4),
                "p_roi_ttest": ptr,
                "p_roi_wilcoxon": pwr,
                "cohen_d_H0_ROI": round(dr, 3),
            }
            raw_cells.append((m, ds))

            vc = {"Clear": 0, "Faint": 0, "Invisible": 0}
            for r in rows:
                vc[r["vis_class"]] += 1
            comp3_summary[m][ds] = {
                "n": len(rows),
                "mean_mu_shift": round(float(np.mean([r["mu_shift"] for r in rows])), 4),
                "mean_var_shift": round(float(np.mean([r["var_shift"] for r in rows])), 4),
                "mean_mse": round(float(np.mean([r["mse"] for r in rows])), 4),
                "mean_psnr_db": round(float(np.mean([r["psnr"] for r in rows])), 2),
                "mean_chi2_hist": round(float(np.mean([r["chi2"] for r in rows])), 2),
                "mean_pov_cov": round(float(np.mean([r["pov_cov"] for r in rows])), 4),
                "mean_pov_stg": round(float(np.mean([r["pov_stg"] for r in rows])), 4),
                "mean_delta_c_bbox": round(float(np.mean([r["delta_c_bbox"] for r in rows])), 2),
                "mean_s_vis": round(float(np.mean([r["s_vis"] for r in rows])), 2),
                "eq9_counts": vc,
                "mean_class": classify_eq9(
                    float(np.mean([r["delta_c_bbox"] for r in rows])),
                    float(np.mean(h0_rs) - np.mean(h0_rc)),
                ),
            }

    pg_t_adj = holm_bonferroni([entropy_summary[m][ds]["p_glob_ttest"] for m, ds in raw_cells])
    pg_w_adj = holm_bonferroni([entropy_summary[m][ds]["p_glob_wilcoxon"] for m, ds in raw_cells])
    pr_t_adj = holm_bonferroni([entropy_summary[m][ds]["p_roi_ttest"] for m, ds in raw_cells])
    pr_w_adj = holm_bonferroni([entropy_summary[m][ds]["p_roi_wilcoxon"] for m, ds in raw_cells])
    for i, (m, ds) in enumerate(raw_cells):
        entropy_summary[m][ds]["p_glob_ttest_holm"] = pg_t_adj[i]
        entropy_summary[m][ds]["p_glob_wilcoxon_holm"] = pg_w_adj[i]
        entropy_summary[m][ds]["p_roi_ttest_holm"] = pr_t_adj[i]
        entropy_summary[m][ds]["p_roi_wilcoxon_holm"] = pr_w_adj[i]

    hex_diff_summary: Dict[str, Any] = {}
    gray_fnames = [rec["clean_fname"] for rec in manifest["records"] if rec["dataset"] == "Gray"]
    for m in METHODS:
        picked_hex = None
        for gf in gray_fnames:
            cov = load_rgb(clean_dir / "Gray" / gf)
            stg = load_rgb(stego_dir / pk0 / m / "Gray" / gf)
            if m in ("LSB", "PVD"):
                c12 = cov[0, :4, :].reshape(-1)
                s12 = stg[0, :4, :].reshape(-1)
                chg = int(np.sum(c12 != s12))
                if m == "LSB" and 4 <= chg <= 8:
                    picked_hex = (gf, "(0,0)", chg, c12, s12)
                    break
                if m == "PVD" and 4 <= chg <= 9 and np.array_equal(c12[3:6], s12[3:6]):
                    picked_hex = (gf, "(0,0)", chg, c12, s12)
                    break
            else:
                ys, xs, _ = np.where(cov != stg)
                if len(ys) > 0:
                    y0 = (int(ys.min()) // 8) * 8
                    x0 = (int(xs.min()) // 8) * 8
                    c12 = cov[y0, x0 : x0 + 4, :].reshape(-1)
                    s12 = stg[y0, x0 : x0 + 4, :].reshape(-1)
                    chg = int(np.sum(c12 != s12))
                    if chg >= 8:
                        picked_hex = (gf, f"({x0},{y0})", chg, c12, s12)
                        break
        assert picked_hex is not None
        gf, coord, chg, c12, s12 = picked_hex
        hex_diff_summary[m] = {
            "sample_fname": gf,
            "coord": coord,
            "changed_bytes": chg,
            "cover_hex": " ".join(f"{v:02X}" for v in c12),
            "stego_hex": " ".join(f"{v:02X}" for v in s12),
        }

    feat_keys = list(clean_res[0]["blind"].keys())
    mode2_summary: Dict[str, Any] = {"Clean": {}, "Stego": {}}
    clean_by_ds = defaultdict(list)
    for r in clean_res:
        clean_by_ds[r["dataset"]].append(r["blind"])

    for ds in DATASETS + ["ALL"]:
        c_list = clean_by_ds[ds] if ds != "ALL" else [r["blind"] for r in clean_res]
        mode2_summary["Clean"][ds] = {
            fk: {
                "p5": round(float(np.percentile([x[fk] for x in c_list], 5)), 4),
                "median": round(float(np.median([x[fk] for x in c_list])), 4),
                "p95": round(float(np.percentile([x[fk] for x in c_list], 95)), 4),
                "p99": round(float(np.percentile([x[fk] for x in c_list], 99)), 4),
                "max": round(float(np.max([x[fk] for x in c_list])), 4),
            }
            for fk in feat_keys
        }

    for pk in PAYLOADS:
        mode2_summary["Stego"][pk] = {}
        for m in METHODS:
            mode2_summary["Stego"][pk][m] = {}
            for ds in DATASETS:
                s_list = [r["blind"] for r in by_pk_m_ds[(pk, m, ds)]]
                mode2_summary["Stego"][pk][m][ds] = {
                    fk: {
                        "p5": round(float(np.percentile([x[fk] for x in s_list], 5)), 4),
                        "median": round(float(np.median([x[fk] for x in s_list])), 4),
                        "p95": round(float(np.percentile([x[fk] for x in s_list], 95)), 4),
                    }
                    for fk in feat_keys
                }

    out_steps = out_dir / "Components_Mode1_Results.json"
    with open(out_steps, "w", encoding="utf-8") as f:
        json.dump(
            {
                "multipayload": multipayload_summary,
                "entropy_comparison": entropy_summary,
                "hex_differential": hex_diff_summary,
                "component3_stats": comp3_summary,
            },
            f,
            indent=2,
        )

    with open(out_dir / "Mode2_Calibration_Stats.json", "w", encoding="utf-8") as f:
        json.dump(mode2_summary, f, indent=2)

    raw_blind = {
        "clean": clean_res,
        "stego": [
            {"pk": r["pk"], "method": r["method"], "dataset": r["dataset"], "fname": r["fname"], "blind": r["blind"]}
            for r in stego_res
        ],
    }
    with open(out_dir / "Mode2_PerImage_Features.json", "w", encoding="utf-8") as f:
        json.dump(raw_blind, f)

    print(f"Saved: {out_steps}")


if __name__ == "__main__":
    main()
