#!/usr/bin/env python3
import argparse
import struct
from torchvision import datasets, transforms


def pack_header(batch_size: int, error_bit: bool = False) -> bytes:
    raw_val = (1 if error_bit else 0) | ((batch_size & 0xFF) << 10)
    return struct.pack("<Q", raw_val)


def generate_raw_mnist(output_path: str, batch_size: int, num_batches: int):
    print("Loading MNIST dataset...")
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    num_dataset_samples = len(dataset)

    # Pre-encode all 60,000 MNIST samples into memory once for fast generation
    print("Pre-encoding samples into byte cache...")
    cache = []
    for img, label in dataset:
        img_tensor = transforms.functional.pil_to_tensor(img)
        sample_bytes = struct.pack("<I", label) + img_tensor.numpy().tobytes()
        cache.append(sample_bytes)

    total_samples = batch_size * num_batches
    print(f"Generating '{output_path}' with {num_batches} batches (total {total_samples} samples)...")

    header_bytes = pack_header(batch_size=batch_size, error_bit=False)

    global_sample_idx = 0
    with open(output_path, "wb") as f:
        for b in range(num_batches):
            f.write(header_bytes)
            for _ in range(batch_size):
                # Cycle infinitely using modulo
                f.write(cache[global_sample_idx % num_dataset_samples])
                global_sample_idx += 1

    file_size_bytes = (8 + (4 + 784) * batch_size) * num_batches
    print(f"Done! Written {num_batches} batches to '{output_path}' ({file_size_bytes / (1024*1024):.2f} MB).")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MNIST binary data.raw with infinite looping support."
    )
    parser.add_argument("-o", "--output", type=str, default="data.raw", help="Output file path")
    parser.add_argument("-b", "--batch-size", type=int, default=60, help="Samples per batch")
    parser.add_argument("-n", "--num-batches", type=int, default=50000, help="Total batches to generate")

    args = parser.parse_args()
    generate_raw_mnist(args.output, args.batch_size, args.num_batches)


if __name__ == "__main__":
    main()