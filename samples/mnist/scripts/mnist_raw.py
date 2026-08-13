#!/usr/bin/env python3
import argparse
import struct
import torch
from torchvision import datasets, transforms


def pack_header(batch_size: int, error_bit: bool = False) -> bytes:
    """
    Packs the header matching the framework's expected header format.
    Adjust the struct layout if your header_t uses a specific bitfield layout.

    Default: uint64_t raw_value with batch_size embedded at bits 10..17.
    """
    # Bit 0: error bit, Bits 10..17: batch_size
    raw_val = (1 if error_bit else 0) | ((batch_size & 0xFF) << 10)
    return struct.pack("<Q", raw_val)  # 8-byte uint64_t (little-endian)


def generate_raw_mnist(output_path: str, batch_size: int, num_batches: int):
    print(f"Loading MNIST dataset...")
    # Load raw MNIST images (0..255 uint8) without normalization transforms
    dataset = datasets.MNIST(root="./data", train=True, download=True)

    total_samples_needed = batch_size * num_batches
    if total_samples_needed > len(dataset):
        raise ValueError(
            f"Requested {total_samples_needed} samples ({num_batches} batches of {batch_size}), "
            f"but test dataset only has {len(dataset)} samples."
        )

    print(f"Generating '{output_path}' with {num_batches} batches (batch_size={batch_size})...")

    written_batches = 0
    with open(output_path, "wb") as f:
        for b in range(num_batches):
            # 1. Write 8-byte Header
            header_bytes = pack_header(batch_size=batch_size, error_bit=False)
            f.write(header_bytes)

            # 2. Write Payload for each sample in the batch
            for i in range(batch_size):
                sample_idx = b * batch_size + i
                img, label = dataset[sample_idx]

                # Convert PIL Image to 784 raw uint8 bytes
                img_tensor = transforms.functional.pil_to_tensor(img)  # Shape: [1, 28, 28], uint8
                pixels_bytes = img_tensor.numpy().tobytes()  # 784 bytes

                # Pack sample: uint32_t label (4 bytes) + 784 uint8 pixels (784 bytes) = 788 bytes
                sample_bytes = struct.pack("<I", label) + pixels_bytes
                f.write(sample_bytes)

            written_batches += 1

    file_size_bytes = (8 + (4 + 784) * batch_size) * num_batches
    print(f"Successfully wrote {written_batches} batches to '{output_path}' ({file_size_bytes} bytes).")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MNIST binary data.raw file with headers for the C++ pipeline."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="data.raw", help="Output file path (default: data.raw)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="Number of samples per batch (default: 1)"
    )
    parser.add_argument(
        "-n", "--num-batches", type=int, default=10, help="Total number of batches to generate (default: 10)"
    )

    args = parser.parse_args()
    generate_raw_mnist(args.output, args.batch_size, args.num_batches)


if __name__ == "__main__":
    main()