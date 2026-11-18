import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

SEED = 3537
DEFAULT_BASE = Path(__file__).resolve().parent.parent

DATASETS = ["SSC_android_icon", "BCC_pokemon", "Gray", "FC_full_color"]
METHODS = ["LSB", "PVD", "BPCS"]

LOREM_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \n"
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \n"
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \n"
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. \n"
    "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, \n"
    "nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. \n"
    "Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper."
)

POW2_DESC = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
SHIFTS_DESC = np.arange(7, -1, -1, dtype=np.uint8)
PVD_MAGIC_HDR = np.array([1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)


def make_structured_binary(target_len: int = 770) -> bytes:
    """Construct a deterministic zero-padded ELF64 + telemetry struct payload."""
    buf = bytearray()
    buf.extend(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8)
    buf.extend(struct.pack("<HHIQQQIHHHHHH", 2, 0x3E, 1, 0x401000, 64, 256, 0, 64, 56, 2, 64, 4, 3))
    rec_idx = 0
    while len(buf) < target_len:
        rec = struct.pack(
            "<IHHII",
            0xCAFE0000 | (rec_idx & 0xFFFF),
            (rec_idx * 17) & 0xFF,
            (rec_idx * 31) & 0xFF,
            rec_idx * 64,
            0x00010003,
        )
        buf.extend(rec)
        rec_idx += 1
    return bytes(buf[:target_len])


def make_compressed_zlib(target_len: int = 770) -> bytes:
    """Construct a raw DEFLATE level-9 compressed byte stream of exact target_len."""
    corpus = (
        LOREM_TEXT
        + " Steganography forensic benchmark evaluation across spatial and transform domains. "
        "Statistical variance, Shannon entropy modeling, and bit-plane anomaly discovery in digital images. "
        "1234567890 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz !@#$%^&*()_+-=[]{}|;':,./<>? "
    ) * 12
    comp_parts = bytearray()
    idx = 0
    while len(comp_parts) < target_len:
        cobj = zlib.compressobj(level=9, wbits=-15)
        chunk = (corpus[idx:] + corpus[:idx] + str(idx)).encode("utf-8")
        comp_parts.extend(cobj.compress(chunk) + cobj.flush())
        idx += 37
    return bytes(comp_parts[:target_len])


def make_encrypted_ctr_keystream(plaintext_bytes: bytes) -> bytes:
    """Encrypt bytes using a deterministic SHA-256 CTR-mode keystream."""
    key = hashlib.sha256(b"SOIC-3537-Forensic-Benchmark-AES256-Key").digest()
    nonce = b"SOIC3537NONC"
    keystream = bytearray()
    counter = 0
    while len(keystream) < len(plaintext_bytes):
        block = hashlib.sha256(key + nonce + struct.pack(">I", counter)).digest()
        keystream.extend(block)
        counter += 1
    return bytes(p ^ k for p, k in zip(plaintext_bytes, keystream[: len(plaintext_bytes)]))


def build_payload_definitions(target_bytes: int = 770) -> Dict[str, Tuple[str, bytes]]:
    b_lorem_full = (LOREM_TEXT * ((target_bytes // len(LOREM_TEXT)) + 2)).encode("utf-8")[:target_bytes]
    b_binary = make_structured_binary(target_bytes)
    b_compressed = make_compressed_zlib(target_bytes)
    b_encrypted = make_encrypted_ctr_keystream(b_lorem_full)
    return {
        "1_Plaintext_LoremIpsum": (f"payload_1_plaintext_lorem_{target_bytes}B.txt", b_lorem_full),
        "2_Structured_Binary": (f"payload_2_structured_binary_{target_bytes}B.bin", b_binary),
        "3_Compressed_Zlib": (f"payload_3_compressed_zlib_{target_bytes}B.bin", b_compressed),
        "4_Encrypted_AES256": (f"payload_4_encrypted_ctr_{target_bytes}B.bin", b_encrypted),
    }


def calc_payload_stats(data_bytes: bytes) -> Dict[str, object]:
    """Compute byte-level Shannon entropy, bit-level Shannon entropy, and 1-bit frequency."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / len(arr)
    h_byte = float(-np.sum(probs * np.log2(probs)))
    bits = np.unpackbits(arr)
    p1 = float(np.mean(bits))
    p0 = 1.0 - p1
    h_bit = float(-(p0 * np.log2(p0) + p1 * np.log2(p1))) if 0 < p1 < 1 else 0.0
    return {
        "bytes": len(data_bytes),
        "bits": int(len(bits)),
        "H_byte": round(h_byte, 4),
        "H_bit": round(h_bit, 4),
        "p1_ratio": round(p1, 4),
        "sha256": hashlib.sha256(data_bytes).hexdigest()[:16],
    }


def inspect_cover_admissibility(args: Tuple[str, str, str]) -> Tuple[str, str, str, int, int, str, bool, int]:
    """Verify minimum dimensions, non-binary content, and tri-encoder single-pass capacity."""
    ds, stratum, path = args
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
        if w < 48 or h < 48:
            return (ds, stratum, path, w, h, mode, False, 0)
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    img_h, img_w, c = arr.shape
    mn, mx = int(arr.min()), int(arr.max())
    if mn == 0 and mx == 255:
        uvals = np.unique(arr)
        if uvals.size <= 2 and set(uvals.tolist()).issubset({0, 255}):
            return (ds, stratum, path, w, h, mode, False, 0)
    elif mn == mx:
        return (ds, stratum, path, w, h, mode, False, 0)

    if img_h * img_w * c < 6248 or (img_w // 3 - 1) * (img_h // 3 - 1) * 24 < 6248:
        return (ds, stratum, path, w, h, mode, False, 0)

    hb, wb = (img_h // 8) * 8, (img_w // 8) * 8
    blocks = arr[:hb, :wb, :].reshape(hb // 8, 8, wb // 8, 8, c).transpose(0, 2, 4, 1, 3).reshape(-1, 8, 8)
    cnt = 0
    for i in range(0, len(blocks), 128):
        chunk = blocks[i : i + 128]
        planes = (chunk[:, None, :, :] >> SHIFTS_DESC[None, :, None, None]) & 1
        hd = np.sum(planes[:, :, :, :-1] != planes[:, :, :, 1:], axis=(2, 3))
        vd = np.sum(planes[:, :, :-1, :] != planes[:, :, 1:, :], axis=(2, 3))
        cnt += int(np.sum((hd + vd) >= 34))
        if cnt >= 97:
            return (ds, f"{img_w}x{img_h}" if ds == "SSC_android_icon" else stratum, path, w, h, mode, True, cnt)
    return (ds, f"{img_w}x{img_h}" if ds == "SSC_android_icon" else stratum, path, w, h, mode, False, cnt)


def hare_niemeyer_quotas(stratum_counts: Dict[str, int], target_n: int = 240) -> Dict[str, int]:
    """Allocate integer proportional quotas using Hare-Niemeyer largest remainders."""
    total = sum(stratum_counts.values())
    exact = {k: (v / total) * target_n for k, v in stratum_counts.items()}
    quotas = {k: int(np.floor(val)) for k, val in exact.items()}
    rem = target_n - sum(quotas.values())
    remainders = sorted(stratum_counts.keys(), key=lambda k: (exact[k] - quotas[k], stratum_counts[k], k), reverse=True)
    for k in remainders:
        if rem <= 0:
            break
        if quotas[k] < stratum_counts[k]:
            quotas[k] += 1
            rem -= 1
    assert sum(quotas.values()) == target_n
    return quotas


def embed_lsb(img_rgb: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
    """Embed bits sequentially in row-major raster order with a 16-bit byte-length header."""
    out = img_rgb.copy()
    flat = out.reshape(-1)
    n_bytes = len(payload_bits) // 8
    header_bits = np.unpackbits(np.array([(n_bytes >> 8) & 0xFF, n_bytes & 0xFF], dtype=np.uint8))
    all_bits = np.concatenate([header_bits, payload_bits])
    n = len(all_bits)
    assert len(flat) >= n, f"LSB capacity overflow: need {n}, have {len(flat)}"
    flat[:n] = (flat[:n] & 0xFE) | all_bits
    return out


def embed_lsb_shuffled(img_rgb: np.ndarray, payload_bits: np.ndarray, seed: int = 3537) -> np.ndarray:
    """Embed bits into PRNG-permuted RGB channel coordinates across the entire cover frame."""
    out = img_rgb.copy()
    flat = out.reshape(-1)
    n_bytes = len(payload_bits) // 8
    header_bits = np.unpackbits(np.array([(n_bytes >> 8) & 0xFF, n_bytes & 0xFF], dtype=np.uint8))
    all_bits = np.concatenate([header_bits, payload_bits])
    n = len(all_bits)
    assert len(flat) >= n, f"Shuffled LSB capacity overflow: need {n}, have {len(flat)}"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat), size=n, replace=False)
    flat[idx] = (flat[idx] & 0xFE) | all_bits
    return out


def embed_pvd_3x3(img_rgb: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
    """Embed bits across 3x3 block corner pixels matching pvd_lib.py range intervals."""
    out = img_rgb.copy()
    h, w, c = out.shape
    n_bytes = len(payload_bits) // 8
    hdr_bytes = np.concatenate(
        [
            PVD_MAGIC_HDR,
            np.array(
                [(n_bytes >> 24) & 0xFF, (n_bytes >> 16) & 0xFF, (n_bytes >> 8) & 0xFF, n_bytes & 0xFF],
                dtype=np.uint8,
            ),
        ]
    )
    all_bits = np.concatenate([np.unpackbits(hdr_bytes), payload_bits])
    total_bits = len(all_bits)
    bit_idx = 0
    corners = ((0, 0), (0, 2), (2, 0), (2, 2))
    no_w = w // 3 - 1
    no_h = h // 3 - 1
    for bx in range(0, no_w * 3, 3):
        for by in range(0, no_h * 3, 3):
            ref_px = out[by + 1, bx + 1]
            for dx, dy in corners:
                for ch in range(c):
                    if bit_idx >= total_bits:
                        return out
                    c_val = int(out[by + dy, bx + dx, ch])
                    diff = abs(c_val - int(ref_px[ch]))
                    if diff < 16:
                        t = 2
                    elif 16 < diff < 32:
                        t = 3
                    else:
                        t = 4
                    seg = all_bits[bit_idx : bit_idx + t]
                    actual_t = len(seg)
                    val = 0
                    for b in seg:
                        val = (val << 1) | int(b)
                    mask = (~((1 << actual_t) - 1)) & 0xFF
                    out[by + dy, bx + dx, ch] = (c_val & mask) | val
                    bit_idx += actual_t
                    if bit_idx >= total_bits:
                        return out
    assert bit_idx >= total_bits, f"PVD capacity overflow: embedded {bit_idx}/{total_bits}"
    return out


def embed_bpcs_single_pass(img_rgb: np.ndarray, payload_bits: np.ndarray, alpha_t: float = 0.30) -> np.ndarray:
    """Embed zero-terminated 8x8 message blocks across MSB-to-LSB planes with complexity >= alpha_t."""
    out = img_rgb.copy()
    h, w, c = out.shape
    with_null = np.concatenate([payload_bits, np.zeros(8, dtype=np.uint8)])
    n_blocks = int(np.ceil(len(with_null) / 64.0))
    padded = np.zeros(n_blocks * 64, dtype=np.uint8)
    padded[: len(with_null)] = with_null
    msg_blocks = padded.reshape(n_blocks, 8, 8)

    msg_idx = 0
    for by in range(0, h - 7, 8):
        for bx in range(0, w - 7, 8):
            for ch in range(c):
                patch = out[by : by + 8, bx : bx + 8, ch]
                planes = (patch[None, :, :] >> SHIFTS_DESC[:, None, None]) & 1
                h_diff = np.sum(planes[:, :, :-1] != planes[:, :, 1:], axis=(1, 2))
                v_diff = np.sum(planes[:, :-1, :] != planes[:, 1:, :], axis=(1, 2))
                alphas = (h_diff + v_diff) / 112.0
                modified = False
                for bp_i in range(8):
                    if alphas[bp_i] >= alpha_t:
                        planes[bp_i] = msg_blocks[msg_idx]
                        msg_idx += 1
                        modified = True
                        if msg_idx >= n_blocks:
                            break
                if modified:
                    out[by : by + 8, bx : bx + 8, ch] = np.sum(
                        planes.astype(np.uint16) * POW2_DESC[:, None, None], axis=0
                    ).astype(np.uint8)
                if msg_idx >= n_blocks:
                    return out
    raise RuntimeError(f"BPCS single-pass overflow: embedded {msg_idx}/{n_blocks} blocks")


def _process_one_cover(item: Tuple[str, str, str, Dict[str, np.ndarray], Path, Path, Path]) -> Tuple[str, str, int, int]:
    ds, clean_fname, src_path, payload_bits_map, clean_out_dir, out_stego_dir, out_reveal_dir = item
    with Image.open(src_path) as im:
        cov = np.array(im.convert("RGB"), dtype=np.uint8)

    clean_path = clean_out_dir / ds / clean_fname
    Image.fromarray(cov).save(clean_path, format="PNG", compress_level=4)

    for pk, p_bits in payload_bits_map.items():
        for method in METHODS:
            if method == "LSB":
                stg = embed_lsb(cov, p_bits)
            elif method == "PVD":
                stg = embed_pvd_3x3(cov, p_bits)
            else:
                stg = embed_bpcs_single_pass(cov, p_bits, alpha_t=0.30)
            rev = ((stg & 1) * 255).astype(np.uint8)
            Image.fromarray(stg).save(out_stego_dir / pk / method / ds / clean_fname, format="PNG", compress_level=4)
            Image.fromarray(rev).save(out_reveal_dir / pk / method / ds / clean_fname, format="PNG", compress_level=4)
    return (ds, clean_fname, cov.shape[1], cov.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified 960-cover cohort and 11,520 multi-payload stego/reveal PNGs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_BASE, help="Workspace root directory")
    args = parser.parse_args()

    base_dir = args.root.resolve()
    real_dir = base_dir / "Real_Dataset"
    cover_dir = real_dir / "1_Cover_Dataset"
    clean_out_dir = real_dir / "7_Unified_Clean_Covers_960"
    out_stego_dir = real_dir / "5_MultiPayload_Stegano"
    out_reveal_dir = real_dir / "6_MultiPayload_Revealed"
    payload_dir = out_stego_dir / "_Payloads_770B"

    t0 = time.time()
    print("=== Step 1: Scanning Cover Pools for Tri-Encoder Single-Pass Admissibility ===")
    candidates: List[Tuple[str, str, str]] = []
    for ds_name, strat in [("SSC_android_icon", "SSC"), ("BCC_pokemon", "BCC"), ("Gray", "Gray")]:
        ds_dir = cover_dir / ds_name
        for f in sorted(os.listdir(ds_dir)):
            if f.lower().endswith(".png"):
                candidates.append((ds_name, strat, str(ds_dir / f)))

    fc_dir = cover_dir / "FC_full_color"
    for sub in sorted(os.listdir(fc_dir)):
        sp = fc_dir / sub
        if sp.is_dir():
            for f in sorted(os.listdir(sp)):
                if f.lower().endswith(".png"):
                    fp = sp / f
                    with Image.open(fp) as im:
                        if im.size == (590, 445):
                            candidates.append(("FC_full_color", sub, str(fp)))

    print(f"Total PNG candidate scan count: {len(candidates)} in {time.time() - t0:.1f}s")
    with ProcessPoolExecutor(max_workers=8) as ex:
        inspected = list(ex.map(inspect_cover_admissibility, candidates, chunksize=64))
    print(f"Admissibility scan completed in {time.time() - t0:.1f}s")

    rng = random.Random(SEED)
    selected_manifest = {}
    pool_summary = {}

    for ds in DATASETS:
        rows = [r for r in inspected if r[0] == ds]
        ok_rows = [r for r in rows if r[6]]
        fail_rows = [r for r in rows if not r[6]]
        strata = defaultdict(list)
        for r in ok_rows:
            strata[r[1]].append(r)
        counts = {k: len(v) for k, v in strata.items()}
        quotas = hare_niemeyer_quotas(counts, 240)

        picked = []
        for k in sorted(strata.keys()):
            q = quotas[k]
            if q > 0:
                pool = sorted(strata[k], key=lambda x: x[2])
                picked.extend(rng.sample(pool, q))
        assert len(picked) == 240, f"{ds} got {len(picked)}"
        picked.sort(key=lambda x: x[2])

        selected_manifest[ds] = picked
        pool_summary[ds] = {
            "scanned": len(rows),
            "admissible": len(ok_rows),
            "excluded": len(fail_rows),
            "strata_pool": counts,
            "strata_sampled": {k: quotas[k] for k in sorted(quotas) if quotas[k] > 0},
        }
        print(f"  {ds}: scanned={len(rows)}, admissible={len(ok_rows)}, excluded={len(fail_rows)}, sampled=240, strata={pool_summary[ds]['strata_sampled']}")

    print("\n=== Step 2: Rebuilding Unified Cohort Directories ===")
    for folder in [clean_out_dir, out_stego_dir, out_reveal_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)

    payload_defs = build_payload_definitions(770)
    for ds in DATASETS:
        (clean_out_dir / ds).mkdir(parents=True, exist_ok=True)
        for pk in payload_defs:
            for method in METHODS:
                (out_stego_dir / pk / method / ds).mkdir(parents=True, exist_ok=True)
                (out_reveal_dir / pk / method / ds).mkdir(parents=True, exist_ok=True)

    payload_meta = {}
    payload_bits_map = {}
    for pk, (fname, pbytes) in payload_defs.items():
        with open(payload_dir / fname, "wb") as f:
            f.write(pbytes)
        payload_meta[pk] = calc_payload_stats(pbytes)
        payload_bits_map[pk] = np.unpackbits(np.frombuffer(pbytes, dtype=np.uint8))

    with open(payload_dir / "README_Payload_Characteristics.json", "w", encoding="utf-8") as f:
        json.dump(payload_meta, f, indent=2)

    work_items = []
    manifest_records = []
    for ds in DATASETS:
        for idx, r in enumerate(selected_manifest[ds], start=1):
            _, stratum, src_path, w, h, native_mode, _, bpcs_cap = r
            safe_strat = stratum.replace(" ", "_")
            base_name = os.path.basename(src_path)
            clean_fname = f"{idx:03d}__{safe_strat}__{base_name}"
            if not clean_fname.lower().endswith(".png"):
                clean_fname = os.path.splitext(clean_fname)[0] + ".png"
            work_items.append((ds, clean_fname, src_path, payload_bits_map, clean_out_dir, out_stego_dir, out_reveal_dir))
            manifest_records.append(
                {
                    "dataset": ds,
                    "idx": idx,
                    "clean_fname": clean_fname,
                    "src_path": os.path.relpath(src_path, str(base_dir)).replace("\\", "/"),
                    "stratum": stratum,
                    "width": w,
                    "height": h,
                    "native_mode": native_mode,
                    "bpcs_blocks_ge_threshold_sample": bpcs_cap,
                }
            )

    print(f"Generating 960 clean covers + 11,520 stego PNGs + 11,520 reveal PNGs ({len(work_items)} covers)...")
    with ProcessPoolExecutor(max_workers=8) as ex:
        list(ex.map(_process_one_cover, work_items, chunksize=8))

    manifest_out = {
        "seed": SEED,
        "pool_summary": pool_summary,
        "payload_meta": payload_meta,
        "records": manifest_records,
    }
    with open(clean_out_dir / "manifest_960.json", "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"=== Completed Unified Cohort Generation in {time.time() - t0:.2f}s ===")


if __name__ == "__main__":
    main()
