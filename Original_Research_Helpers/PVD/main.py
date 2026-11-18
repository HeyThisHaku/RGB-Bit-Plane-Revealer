import argparse
import os
import subprocess
import sys


def process_cover(cover_path: str, output_folder: str, filename: str, secret_file: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "test_main.py"), "E", cover_path, secret_file, output_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 3x3 block PVD embedding helper.")
    parser.add_argument("--cover-root", default="./cover", help="Root folder containing cover PNGs")
    parser.add_argument("--secret", default="secret.txt", help="Secret payload file path")
    parser.add_argument("--output-root", default="./stego_pvd", help="Output folder")
    args = parser.parse_args()

    for root, _, files in os.walk(args.cover_root):
        rel_path = os.path.relpath(root, args.cover_root)
        output_folder = os.path.join(args.output_root, rel_path)
        png_files = [f for f in sorted(files) if f.lower().endswith(".png")]
        if not png_files:
            continue
        print(f"Folder {rel_path}: {len(png_files)} PNG file(s)")
        for file in png_files:
            process_cover(os.path.join(root, file), output_folder, file, args.secret)
    print("Done.")


if __name__ == "__main__":
    main()
