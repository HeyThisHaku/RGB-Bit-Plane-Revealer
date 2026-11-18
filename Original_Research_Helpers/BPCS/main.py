import argparse
import os
import cv2
from bpcs import BPCS
from message import Message


def process_file(input_path: str, output_path: str, secret_text: str) -> None:
    try:
        bpcs = BPCS(input_path)
        bitplane_msg = Message.create_message(secret_text)
        img_result = bpcs.hide(bitplane_msg)
        cv2.imwrite(output_path, img_result)
        print(f"Saved: {output_path}")
    except Exception as exc:
        print(f"Failed processing {os.path.basename(input_path)}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch BPCS embedding helper.")
    parser.add_argument("--cover-root", default="./cover", help="Root folder containing cover PNGs")
    parser.add_argument("--output-dir", default="./stego_bpcs", help="Output directory")
    parser.add_argument("--secret", default="secret.txt", help="Path to UTF-8/ASCII secret message file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.secret, "r", encoding="utf-8") as f:
        secret_text = f.read()

    folders = [f for f in sorted(os.listdir(args.cover_root)) if os.path.isdir(os.path.join(args.cover_root, f))]
    if folders:
        print("Detected subfolders:", folders)
        for folder in folders:
            folder_path = os.path.join(args.cover_root, folder)
            files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            print(f"Folder {folder}: {len(files)} file(s)")
            for filename in files:
                process_file(
                    os.path.join(folder_path, filename),
                    os.path.join(args.output_dir, f"{folder}_{filename}"),
                    secret_text,
                )
    else:
        files = [f for f in sorted(os.listdir(args.cover_root)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"Processing {len(files)} file(s) from root folder")
        for filename in files:
            process_file(
                os.path.join(args.cover_root, filename),
                os.path.join(args.output_dir, filename),
                secret_text,
            )
    print("Done.")


if __name__ == "__main__":
    main()
