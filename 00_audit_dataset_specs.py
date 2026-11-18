import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = Path(__file__).resolve().parent.parent
SHIFTS_DESC = np.arange(7, -1, -1, dtype=np.uint8)
DATASETS = ["SSC_android_icon", "BCC_pokemon", "Gray", "FC_full_color"]
METHODS = ["LSB", "PVD", "BPCS"]


def count_bpcs_complex_blocks(rgb_arr: np.ndarray, alpha_t: float = 0.30) -> int:
    """Count 8x8 bit-plane blocks with boundary-transition complexity >= alpha_t."""
    h, w, c = rgb_arr.shape
    hb, wb = (h // 8) * 8, (w // 8) * 8
    if hb < 8 or wb < 8:
        return 0
    blks = (
        rgb_arr[:hb, :wb, :]
        .reshape(hb // 8, 8, wb // 8, 8, c)
        .transpose(0, 2, 4, 1, 3)
        .reshape(-1, 8, 8)
    )
    planes = (blks[:, None, :, :] >> SHIFTS_DESC[None, :, None, None]) & 1
    hd = np.sum(planes[:, :, :, :-1] != planes[:, :, :, 1:], axis=(2, 3))
    vd = np.sum(planes[:, :, :-1, :] != planes[:, :, 1:, :], axis=(2, 3))
    alphas = (hd + vd) / 112.0
    return int(np.sum(alphas >= alpha_t))


def _pvd_boundary_one(args: Tuple[Path, int]) -> Tuple[int, int, int, int]:
    p, total_bits = args
    with Image.open(p) as im:
        arr = np.array(im.convert("RGB"), dtype=np.int16)
    h, w, c = arr.shape
    no_w, no_h = w // 3 - 1, h // 3 - 1
    corners = ((0, 0), (0, 2), (2, 0), (2, 2))
    c_lt16 = c_eq16 = c_mid = c_ge32 = 0
    bit_idx = 0
    done = False
    for bx in range(0, no_w * 3, 3):
        for by in range(0, no_h * 3, 3):
            ref = arr[by + 1, bx + 1]
            for dx, dy in corners:
                for ch in range(c):
                    if bit_idx >= total_bits:
                        done = True
                        break
                    diff = abs(int(arr[by + dy, bx + dx, ch]) - int(ref[ch]))
                    if diff < 16:
                        t = 2
                        c_lt16 += 1
                    elif diff == 16:
                        t = 4
                        c_eq16 += 1
                    elif diff < 32:
                        t = 3
                        c_mid += 1
                    else:
                        t = 4
                        c_ge32 += 1
                    bit_idx += t
                if done:
                    break
            if done:
                break
        if done:
            break
    return c_lt16, c_eq16, c_mid, c_ge32


def audit_pvd_boundary_distribution(clean_dir: Path, records: list[dict[str, Any]], payload_bytes: int = 770) -> Dict[str, Any]:
    """Measure exact center-corner difference counts including d == 16 boundary rate."""
    total_bits = (payload_bytes + 11) * 8
    tasks = [(clean_dir / rec["dataset"] / rec["clean_fname"], total_bits) for rec in records]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_pvd_boundary_one, tasks, chunksize=32))
    cnt_lt16 = sum(r[0] for r in res)
    cnt_eq16 = sum(r[1] for r in res)
    cnt_17_31 = sum(r[2] for r in res)
    cnt_ge32 = sum(r[3] for r in res)
    tot = cnt_lt16 + cnt_eq16 + cnt_17_31 + cnt_ge32
    return {
        "total_embedded_corners": tot,
        "lt_16": cnt_lt16,
        "eq_16": cnt_eq16,
        "gt_16_lt_32": cnt_17_31,
        "ge_32": cnt_ge32,
        "pct_eq_16": round(cnt_eq16 / max(1, tot) * 100.0, 4),
    }


def _stat_one_header(p: Path) -> Tuple[int, str, str, str]:
    with Image.open(p) as im:
        return p.stat().st_size, os.path.splitext(p.name.lower())[1], f"{im.size[0]}x{im.size[1]}", im.mode


def scan_raw_dataset_sizes(folder: Path, exts: tuple[str, ...]) -> Dict[str, Any]:
    """Scan raw image pool populations, file sizes, color modes, and resolutions."""
    paths: List[Path] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(Path(root) / fn)
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_stat_one_header, paths, chunksize=64))
    bytes_list = [r[0] for r in rows]
    ext_counter: Counter[str] = Counter(r[1] for r in rows)
    res_counter: Counter[str] = Counter(r[2] for r in rows)
    mode_counter: Counter[str] = Counter(r[3] for r in rows)
    mean_b = float(np.mean(bytes_list))
    return {
        "n_files": len(bytes_list),
        "mean_kb_decimal": round(mean_b / 1000.0, 2),
        "mean_kib_binary": round(mean_b / 1024.0, 2),
        "extensions": dict(ext_counter),
        "modes": dict(mode_counter),
        "resolutions_top10": dict(res_counter.most_common(10)),
        "min_res": min(res_counter.keys(), key=lambda s: tuple(int(x) for x in s.split("x"))),
        "max_res": max(res_counter.keys(), key=lambda s: tuple(int(x) for x in s.split("x"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw datasets, unified cohort capacity, and archival inventories.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    raw_dir = base_dir / "Real_Dataset" / "1_Cover_Dataset"
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    arch_stg_dir = base_dir / "Real_Dataset" / "3_Stegano_Sampled_240"
    arch_rev_dir = base_dir / "Real_Dataset" / "4_Revealed_Sampled_240"
    reveal_photo_dir = base_dir / "Paper_LaTeX" / "reveal_photo"
    out_path = base_dir / "Code" / "output_metrics" / "Dataset_Specs_Summary.json"

    t0 = time.time()
    raw_summary = {
        ds: scan_raw_dataset_sizes(raw_dir / ds, (".png", ".jpg", ".jpeg"))
        for ds in DATASETS
    }

    with Image.open(raw_dir / "Gray" / "5906.png") as im:
        blocks_5906 = count_bpcs_complex_blocks(np.array(im.convert("RGB"), dtype=np.uint8))

    fc_species = {}
    fc_root = raw_dir / "FC_full_color"
    for sub in sorted(os.listdir(fc_root)):
        sp = fc_root / sub
        if sp.is_dir():
            pngs = sum(1 for f in os.listdir(sp) if f.lower().endswith(".png"))
            jpgs = sum(1 for f in os.listdir(sp) if f.lower().endswith((".jpg", ".jpeg")))
            fc_species[sub] = {"png": pngs, "jpg": jpgs}

    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        man = json.load(f)

    unified_summary = {}
    for ds in DATASETS:
        recs = [r for r in man["records"] if r["dataset"] == ds]
        f_bytes = [(clean_dir / ds / r["clean_fname"]).stat().st_size for r in recs]
        bpps = [6160.0 / (r["width"] * r["height"]) for r in recs]
        cov_pcts = [6160.0 / (3.0 * r["width"] * r["height"]) * 100.0 for r in recs]
        unified_summary[ds] = {
            "n_sampled": len(recs),
            "normalized_mean_kib": round(float(np.mean(f_bytes)) / 1024.0, 2),
            "mean_bpp_6160b": round(float(np.mean(bpps)), 4),
            "mean_channel_cov_pct": round(float(np.mean(cov_pcts)), 2),
            "pool_summary": man["pool_summary"][ds],
        }

    pvd_boundary_audit = audit_pvd_boundary_distribution(clean_dir, man["records"], payload_bytes=770)

    archival: Dict[str, Any] = {}
    for method in METHODS:
        archival[method] = {}
        for ds in DATASETS:
            sp = arch_stg_dir / method / ds
            rp = arch_rev_dir / method / ds
            n_s = len([f for f in os.listdir(sp) if f.lower().endswith(".png")]) if sp.is_dir() else 0
            n_r = len([f for f in os.listdir(rp) if f.lower().endswith(".png")]) if rp.is_dir() else 0
            archival[method][ds] = {"stego_count": n_s, "reveal_count": n_r}

    reveal_figs = []
    if reveal_photo_dir.is_dir():
        for root, _, files in os.walk(reveal_photo_dir):
            for fn in sorted(files):
                if fn.lower().endswith(".png"):
                    p = Path(root) / fn
                    rel = os.path.relpath(str(p), str(reveal_photo_dir)).replace("\\", "/")
                    with Image.open(p) as im:
                        reveal_figs.append({"path": rel, "size": list(im.size), "mode": im.mode})

    out = {
        "raw_population_summary": raw_summary,
        "gray_5906_bpcs_blocks": blocks_5906,
        "fc_species_counts": fc_species,
        "unified_960_summary": unified_summary,
        "pvd_boundary_d16_audit": pvd_boundary_audit,
        "archival_sampled_240_counts": archival,
        "paper_reveal_photo_figures": reveal_figs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Dataset and archival audit completed in {time.time() - t0:.2f}s")
    print(f"Gray/5906.png complex blocks: {blocks_5906}")
    print(f"PVD boundary audit (d == 16): {pvd_boundary_audit}")
    for ds, u in unified_summary.items():
        r = raw_summary[ds]
        print(
            f"  {ds:16s} | N={r['n_files']:5d} | Raw={r['mean_kb_decimal']:.2f} KB ({r['mean_kib_binary']:.2f} KiB) "
            f"| Norm={u['normalized_mean_kib']:.2f} KiB | bpp={u['mean_bpp_6160b']:.4f} ({u['mean_channel_cov_pct']:.2f}%)"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
