# RGB-Bit-Plane-Revealer

Reproducible Python 3.11 research codebase accompanying the manuscript:

> **Revealing Hidden Messages: A Visual-Aided Approach for Steganography Detection in Image Forensic**  
> Aditya Kurniawan, Sandyka Bala\*, and Hans Indrawan Sucipto  
> School of Computer Science, Bina Nusantara University, Jl. Raya Kebon Jeruk No. 27, Palmerah, Jakarta Barat 11530, Indonesia  
> \*Corresponding author: `sandyka.bala@binus.edu` — *Statistics, Optimization & Information Computing (SOIC)* (`DOI: 10.19139/soic-2310-5070-3537`)  
> Repository: [https://github.com/HeyThisHaku/RGB-Bit-Plane-Revealer](https://github.com/HeyThisHaku/RGB-Bit-Plane-Revealer)

---

## 1. Framework Overview

The framework bridges visual bit-plane inspection and quantitative statistical steganalysis across two operating modes and three analytical layers:

1. **Component 1 — RGB Bit-Plane Revealer:** Projects 0-th bit-plane intensity anomalies (`(pixel & 1) * 255`), isolating origin-anchored top scanlines (sequential LSB), vertical $3\times 3$ left-margin corner clusters ($3\times 3$ PVD), and complex $8\times 8$ foreground texture substitutions (BPCS, $\alpha \ge 0.30$).
2. **Component 2 — Hexadecimal Color & Entropy Analyzer:** Extracts byte-level hexadecimal diffs around $(0,0)$, global byte Shannon entropy ($H_{\text{global}}$), ROI LSB-plane entropy ($H_{0,\text{ROI}}$), upper-strip Fridrich RS fill rate ($\hat{p}_{\text{RS,top}}$), and BPCS MSB column-0 parity asymmetry ($\Delta_{\text{col0}}$).
3. **Component 3 — Dataset Statistical Analyzer:** Quantifies global $|\Delta\mu|$, $|\Delta\sigma^2|$, MSE, PSNR, 256-bin histogram $\chi^2_{\text{hist}}$, Pair-of-Values asymmetry ($\text{PoV}_{\text{asym}}$), objective contrast visibility (Eq. 9), and block spatial uncorrelation / plane-monotonicity inversion.

| Operational Mode | Reference Required | Calibration | Verified Cohort Performance ($N=12{,}480$) |
| :--- | :--- | :--- | :--- |
| **Mode I (Paired)** | Trusted Cover $I_c$ | Deterministic Eqs. (1)–(9) | **100.00% Recall** (11,520/11,520 stego), **0.00% Clean FPR** (0/960) |
| **Mode II (Blind)** | None ($I_s$ only) | **5-Fold Cover-Grouped CV** | **92.60% Acc**, **93.94% BalAcc**, **0.9669 AUC**, **4.48% Clean FPR** (`[3.34%, 5.98%]`) |

---

## 2. Repository Structure

| Path | Stage / Purpose |
| :--- | :--- |
| `00_audit_dataset_specs.py` | Stage 1: Raw image pools, capacity admissibility filtering, and PVD `d=16` boundary audit |
| `01_build_unified_cohort.py` | Stage 2: Stratified `seed=3537` curation (960 covers) and 11,520 P1–P4 stego embeddings |
| `02_extract_components_and_mode1.py` | Stage 3: Mode I paired metrics and payload-agnostic Mode II feature extraction |
| `03_evaluate_mode2_cv.py` | Stage 4: Strict 5-fold cover-grouped out-of-fold CV evaluation for Mode II |
| `04_extract_baselines_and_spam.py` | Stage 5: Westfeld $\chi^2$, Fridrich RS, Dumitrescu SPA, and SPAM-686 extraction + dense LSB sanity check |
| `05_run_forensic_benchmarks.py` | Stage 6: Ablation, SVM CV, McNemar test, variable rates, PRNG LSB, 8 distortion regimes, AC1, and runtime/RAM |
| `06_generate_figures.py` | Stage 7: Publication 300-DPI PNG figure generator (ROC comparison, entropy boxplots, NN reveal grid) |
| `figure1_flowchart.tex` | Standalone TikZ source for Manuscript Figure 1 |
| `Original_Research_Helpers/` | Clean English, comment-free encoder and reveal reference helpers (`revealRGB.py`, `LSB/`, `PVD/`, `BPCS/`) |
| `output_metrics/` | Precomputed lightweight summary JSON logs and ROC curve cache |
| `requirements.txt` | Pinned Python 3.11 dependencies |
| `LICENSE` | MIT License |

---

## 3. Environment Setup & Execution

Tested on Windows 10/11 with **Python 3.11.5**. Clone the repository and install pinned dependencies:

```bash
git clone https://github.com/HeyThisHaku/RGB-Bit-Plane-Revealer.git Code
cd Code
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Expected Workspace Layout

Every pipeline script accepts `--root <workspace_dir>` (defaulting to the parent directory `..` when cloned into `<workspace>/Code/`). Place the benchmark collections (`SSC_android_icon`, `BCC_pokemon`, `Gray` BOSSBase 1.01, and `FC_full_color` Large-Scale Fish Dataset) under `<workspace>/Real_Dataset/1_Cover_Dataset/` prior to running Stages 1–2, or point `--root` to an existing workspace directory containing `Real_Dataset/`:

```bash
python 00_audit_dataset_specs.py --root ..
python 01_build_unified_cohort.py --root ..
python 02_extract_components_and_mode1.py --root ..
python 03_evaluate_mode2_cv.py --root .. --from-cache
python 04_extract_baselines_and_spam.py --root ..
python 05_run_forensic_benchmarks.py --root ..
python 06_generate_figures.py --root ..
```

---

## 4. Key Quantitative Results (`seed=3537`, Out-of-Fold CV)

* **5-Fold Cover-Grouped Blind Ablation ($N = 12{,}480$):** Component 1 alone (`55.22%` Rec, `76.93%` BalAcc), Component 2 alone (`34.71%` Rec, `66.73%` BalAcc), and Component 3 alone (`34.85%` Rec, `66.33%` BalAcc) fuse in Mode II to reach **`92.35%` Rec** (`[91.85%, 92.82%]`), **`93.94%` BalAcc**, **`0.9669` AUC**, and **`4.48%` Clean FPR** (`[3.34%, 5.98%]`).
* **Blind Baselines (`Table 12`):** Mode II (`93.94%` BalAcc) outperforms SPAM-686 + RBF-SVM (`76.22%` BalAcc, `26.88%` FPR; paired McNemar $\chi^2_{\text{cc}} = 906.55$, $p = 1.65\times 10^{-209}$), Dumitrescu SPA (`54.92%` BalAcc, `17.50%` FPR), Fridrich RS (`51.57%`), and Westfeld $\chi^2$ (`50.01%`–`50.59%`).
* **Variable Sequential Rates & Operational Boundaries (`Table 13`):** Mode II detects `96.88%`–`100.00%` of sequential LSB across `256`–`2,048` B and `100.00%` of BPCS at $\ge 512$ B; when LSB is scattered uniformly via PRNG key permutation (`LSB_Shuffled_770B`), origin-anchored Mode II drops to `9.90%` while SPAM-SVM retains `87.92%`.
* **Post-Processing & Spatial Distortions (`Table 14`):** Mode II retains the highest Balanced Accuracy at JPEG $Q=95$ (`70.73%`), AWGN $\sigma=1$ (`65.05%`), 8 px crop (`63.04%`), and $0.9\times$ bicubic resize (`77.10%`) while preserving low clean FPR compared with SPAM-SVM and SPA.
* **Single-Threaded $\mathcal{O}(W\cdot H)$ Throughput (`Table 15`):** `57.87 +/- 5.05 ms` (`13.81 MB` peak heap) on $590\times 445$ photography — **`3.58x` faster than SPAM-686 + SVM** (`206.96 ms`) and **`1.75x` faster than Fridrich RS** (`101.35 ms`).

---

## 5. License

Released under the [MIT License](LICENSE).
