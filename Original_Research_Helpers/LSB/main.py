import argparse
import os
import cv2
import LSBSteg

SECRET_TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \n"
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \n"
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \n"
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. \n"
    "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, \n"
    "nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. \n"
    "Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper."
)


def process_file(input_path: str, output_path: str, secret_text: str) -> None:
    filename = os.path.basename(input_path)
    print(f"Processing: {filename}")
    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to read image: {filename}")
        return
    steg = LSBSteg.LSBSteg(image)
    img_encoded = steg.encode_text(secret_text)
    cv2.imwrite(output_path, img_encoded)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch LSB embedding helper.")
    parser.add_argument("--cover-root", default="./cover", help="Root folder containing cover PNGs")
    parser.add_argument("--output-root", default="./stego", help="Root output directory")
    parser.add_argument("--output-subdir", default="lsb", help="Subdirectory under output-root")
    args = parser.parse_args()

    for root, _, files in os.walk(args.cover_root):
        rel_path = os.path.relpath(root, args.cover_root)
        output_folder = os.path.join(args.output_root, args.output_subdir, rel_path)
        os.makedirs(output_folder, exist_ok=True)
        images = [f for f in sorted(files) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not images:
            continue
        print(f"Folder {rel_path}: {len(images)} image(s)")
        for filename in images:
            process_file(os.path.join(root, filename), os.path.join(output_folder, filename), SECRET_TEXT)
    print("Done.")


if __name__ == "__main__":
    main()
