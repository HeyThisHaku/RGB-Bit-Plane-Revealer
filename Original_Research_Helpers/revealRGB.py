from pathlib import Path
import argparse
import numpy as np
from PIL import Image


def extract_lsb_rgb_visual(input_path: Path, output_path: Path) -> None:
    """Project the 0-th bit plane of each RGB channel into [0, 255]."""
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    lsb_image = ((img_array & 1) * 255).astype(np.uint8)
    Image.fromarray(lsb_image).save(output_path)
    print(f"Saved LSB reveal visualization: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RGB 0-th bit-plane reveal visualizations.")
    parser.add_argument("--input-root", type=Path, default=Path("./stego"), help="Input image folder")
    parser.add_argument("--output-root", type=Path, default=Path("./revealed"), help="Output folder")
    args = parser.parse_args()

    for root, _, files in Path(args.input_root).walk() if hasattr(Path, "walk") else __import__("os").walk(args.input_root):
        rel_path = __import__("os").path.relpath(str(root), str(args.input_root))
        output_folder = Path(args.output_root) / rel_path
        output_folder.mkdir(parents=True, exist_ok=True)
        images = sorted(f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not images:
            continue
        print(f"Processing folder {rel_path}: {len(images)} image(s)")
        for filename in images:
            extract_lsb_rgb_visual(Path(root) / filename, output_folder / filename)
    print("Done.")


if __name__ == "__main__":
    main()
