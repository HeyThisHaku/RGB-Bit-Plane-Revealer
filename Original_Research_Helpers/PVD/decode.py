from pvd_lib import pvd_lib


def decode():
    pvd_obj = pvd_lib()
    cover_image = "target.png"
    stego_image = "target_pvd.png"
    output_secret = "decoded_secret.txt"

    print(f"Decoding {stego_image} using {cover_image}...")
    try:
        pvd_obj.pvd_extract(cover_image, output_secret, stego_image)
        print(f"Success! Decoded secret saved to: {output_secret}")
        with open(output_secret, "r", encoding="utf-8", errors="replace") as f:
            print("Decoded Secret Content:")
            print("-" * 20)
            print(f.read())
            print("-" * 20)
    except Exception as e:
        print(f"Error during decoding: {e}")


if __name__ == "__main__":
    decode()
