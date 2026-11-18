import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = Path(__file__).resolve().parent.parent
DATASETS = ["SSC_android_icon", "BCC_pokemon", "Gray", "FC_full_color"]
DS_LABELS = {"SSC_android_icon": "SSC", "BCC_pokemon": "BCC", "Gray": "Gray", "FC_full_color": "FC"}
METHODS = ["LSB", "PVD", "BPCS"]


def _load_mod02(code_dir: Path):
    spec = importlib.util.spec_from_file_location("mod_02", str(code_dir / "02_extract_components_and_mode1.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def plot_roc_comparison(out_dir: Path, fig_dir: Path) -> None:
    """Plot exact ROC curves comparing Mode II (5-fold CV) against literature baselines."""
    roc_path = out_dir / "ROC_Curves_Data.npz"
    if not roc_path.exists():
        raise FileNotFoundError(f"Missing ROC data cache: {roc_path}")
    d = np.load(roc_path)

    curves = [
        ("Proposed_Mode_II_5FoldGroupCV", "Proposed Mode II (5-Fold Group CV)", "#0B5394", "-", 2.4),
        ("Pevny_SPAM686_RBF_SVM_5FoldGroupCV", "Pevny SPAM-686 + RBF-SVM (5-Fold CV)", "#C0392B", "--", 2.0),
        ("Fridrich_RS_Full_r005", "Fridrich RS Analysis (RGB Mean)", "#1E8449", "-.", 1.8),
        ("Dumitrescu_SPA_Full_r005", "Dumitrescu Sample Pair Analysis (SPA)", "#D68910", ":", 2.0),
        ("Westfeld_Chi2_Prefix10_p095", "Westfeld Chi-Square (Prefix 10%)", "#7D3C98", (0, (3, 1, 1, 1)), 1.6),
        ("Westfeld_Chi2_Full_p095", "Westfeld Chi-Square (Full Image)", "#566573", (0, (1, 2)), 1.4),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=300)
    for key, label, color, ls, lw in curves:
        fpr = d[f"{key}_fpr"]
        tpr = d[f"{key}_tpr"]
        auc_v = float(d[f"{key}_auc"][0])
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc_v:.4f})", color=color, linestyle=ls, linewidth=lw)

    ax.plot([0, 1], [0, 1], color="#99A3A4", linestyle="--", linewidth=1.0, label="Chance Baseline (AUC = 0.5000)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False Positive Rate (Clean Covers Misclassified)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Stego Recall)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Blind Steganalysis ROC Comparison Across Unified Multi-Domain Cohort\n(N = 12,480 Evaluations; Seed = 3537)",
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92)
    fig.tight_layout()
    out_file = fig_dir / "fig_roc_comparison.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_entropy_distributions(base_dir: Path, fig_dir: Path) -> None:
    """Plot ROI LSB-plane Shannon entropy (H_0,ROI) boxplots comparing Cover vs Stego P1."""
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    stego_dir = base_dir / "Real_Dataset" / "5_MultiPayload_Stegano" / "1_Plaintext_LoremIpsum"
    mod02 = _load_mod02(base_dir / "Code")

    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        manifest = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), dpi=300, sharey=True)
    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        cov_groups: List[List[float]] = []
        stg_groups: List[List[float]] = []
        for ds in DATASETS:
            recs = [r for r in manifest["records"] if r["dataset"] == ds]
            c_vals, s_vals = [], []
            for r in recs:
                cov = mod02.load_rgb(clean_dir / ds / r["clean_fname"])
                stg = mod02.load_rgb(stego_dir / method / ds / r["clean_fname"])
                res = mod02.compute_mode1_pair_array(cov, stg, method=method, compute_comp3=False)
                c_vals.append(res["h0_roi_cov"])
                s_vals.append(res["h0_roi_stg"])
            cov_groups.append(c_vals)
            stg_groups.append(s_vals)

        pos_c = np.arange(len(DATASETS)) * 2.2 - 0.35
        pos_s = np.arange(len(DATASETS)) * 2.2 + 0.35

        bp_c = ax.boxplot(
            cov_groups,
            positions=pos_c,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#1B4F72", linewidth=1.4),
        )
        bp_s = ax.boxplot(
            stg_groups,
            positions=pos_s,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#7B241C", linewidth=1.4),
        )
        for box in bp_c["boxes"]:
            box.set(facecolor="#AED6F1", edgecolor="#1B4F72", linewidth=1.0)
        for box in bp_s["boxes"]:
            box.set(facecolor="#F5B7B1", edgecolor="#7B241C", linewidth=1.0)

        ax.set_xticks(np.arange(len(DATASETS)) * 2.2)
        ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=9.5, fontweight="bold")
        ax.set_title(f"{method} Embedding Locus", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.03, 1.06)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        if ax_idx == 0:
            ax.set_ylabel(r"Local ROI Shannon Entropy ($H_{0,\mathrm{ROI}}$, bits/bit)", fontsize=9.5, fontweight="bold")
        if ax_idx == 2:
            ax.legend([bp_c["boxes"][0], bp_s["boxes"][0]], ["Clean Cover", "Stego (P1)"], loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle("Cover vs. Stego ROI Bit-Plane Entropy Distributions Across Visual Domains (n = 240/group)", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_file = fig_dir / "fig_entropy_distributions.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_crisp_reveal_panel(base_dir: Path, fig_dir: Path) -> None:
    """Render 4x5 nearest-neighbor visual verification grid including clean-cover controls."""
    clean_dir = base_dir / "Real_Dataset" / "7_Unified_Clean_Covers_960"
    stego_dir = base_dir / "Real_Dataset" / "5_MultiPayload_Stegano" / "1_Plaintext_LoremIpsum"
    rev_dir = base_dir / "Real_Dataset" / "6_MultiPayload_Revealed" / "1_Plaintext_LoremIpsum"

    with open(clean_dir / "manifest_960.json", encoding="utf-8") as f:
        manifest = json.load(f)

    picked = {}
    for ds in DATASETS:
        cands = [r for r in manifest["records"] if r["dataset"] == ds]
        if ds == "SSC_android_icon":
            c64 = [r for r in cands if r["width"] == 64 and r["height"] == 64]
            picked[ds] = c64[len(c64) // 2]["clean_fname"]
        else:
            picked[ds] = cands[0]["clean_fname"]

    col_titles = [
        "Clean Cover\n(Original RGB)",
        "Reveal Control\n(Clean LSB x255)",
        "Reveal LSB\n(Top Scanline Band)",
        "Reveal 3x3 PVD\n(Left Column Corners)",
        "Reveal BPCS\n(Complex 8x8 Tiles)",
    ]

    fig, axes = plt.subplots(len(DATASETS), 5, figsize=(11.5, 9.0), dpi=300)
    for r_idx, ds in enumerate(DATASETS):
        fn = picked[ds]
        with Image.open(clean_dir / ds / fn) as im:
            cov = np.array(im.convert("RGB"), dtype=np.uint8)
        rev_clean = ((cov & 1) * 255).astype(np.uint8)
        imgs = [cov, rev_clean]
        for method in METHODS:
            with Image.open(rev_dir / method / ds / fn) as rm:
                imgs.append(np.array(rm.convert("RGB"), dtype=np.uint8))

        for c_idx, arr in enumerate(imgs):
            ax = axes[r_idx, c_idx]
            ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#7F8C8D")
                spine.set_linewidth(0.8)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.0, fontweight="bold", pad=6)
            if c_idx == 0:
                h, w, _ = cov.shape
                ax.set_ylabel(f"{DS_LABELS[ds]}\n({w}x{h})", fontsize=9.5, fontweight="bold", rotation=90, labelpad=6)

    fig.suptitle(
        "Nearest-Neighbor Bit-Plane Reveal Panel: Clean Controls vs. Multi-Method Stego Footprints (Seed = 3537)",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    out_file = fig_dir / "fig_reveal_panel_nn.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures (ROC, entropy boxplots, nearest-neighbor reveal grid).")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    out_dir = base_dir / "Code" / "output_metrics"
    fig_dir = base_dir / "Paper_LaTeX" / "reveal_photo"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_comparison(out_dir, fig_dir)
    plot_entropy_distributions(base_dir, fig_dir)
    plot_crisp_reveal_panel(base_dir, fig_dir)


if __name__ == "__main__":
    main()
