"""
Convert local grid .npz files to .h5 format for faster data loading.

This script converts individual .npz files to .h5 format, which provides:
- Faster I/O operations
- Better compression
- Memory-mapped access without loading entire file

Usage:
    python scripts/convert_npz_to_h5.py --data_dir <path_to_npz_dir> [--delete_npz] [--num_workers 4]
"""

import os
import os.path as osp
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from functools import partial


def convert_single_file(npz_path, delete_npz=False, compression_level=4):
    """
    Convert a single .npz file to .h5 format.

    Args:
        npz_path: Path to the .npz file
        delete_npz: Whether to delete the .npz file after conversion
        compression_level: gzip compression level (0-9, higher = better compression but slower)

    Returns:
        tuple: (npz_path, h5_path, success, error_msg)
    """
    h5_path = npz_path.replace('.npz', '.h5')

    try:
        # Load .npz file
        data = np.load(npz_path)

        # Create .h5 file and write data
        with h5py.File(h5_path, 'w') as f:
            for key in data.files:
                f.create_dataset(
                    key,
                    data=data[key],
                    compression='gzip',
                    compression_opts=compression_level
                )

        # Optionally delete original .npz file
        if delete_npz:
            os.remove(npz_path)

        return (npz_path, h5_path, True, None)

    except Exception as e:
        return (npz_path, h5_path, False, str(e))


def convert_directory(data_dir, delete_npz=False, num_workers=4, compression_level=4):
    """
    Convert all .npz files in a directory to .h5 format.

    Args:
        data_dir: Directory containing .npz files
        delete_npz: Whether to delete .npz files after successful conversion
        num_workers: Number of parallel workers
        compression_level: gzip compression level (0-9)
    """
    # Find all .npz files
    npz_files = sorted(glob(osp.join(data_dir, '*.npz')))

    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} .npz files in {data_dir}")
    print(f"Using {num_workers} workers")
    print(f"Compression level: {compression_level}")
    print(f"Delete original .npz files: {delete_npz}")

    # Create conversion function with fixed parameters
    convert_fn = partial(
        convert_single_file,
        delete_npz=delete_npz,
        compression_level=compression_level
    )

    # Process files in parallel
    failed_files = []
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_fn, npz_files),
            total=len(npz_files),
            desc="Converting files"
        ))

    # Check for failures
    for npz_path, h5_path, success, error_msg in results:
        if not success:
            failed_files.append((npz_path, error_msg))

    # Print summary
    print(f"\nConversion complete!")
    print(f"Successfully converted: {len(npz_files) - len(failed_files)}/{len(npz_files)}")

    if failed_files:
        print(f"\nFailed conversions ({len(failed_files)}):")
        for npz_path, error_msg in failed_files:
            print(f"  {npz_path}: {error_msg}")


def verify_conversion(npz_path, h5_path):
    """
    Verify that .h5 file contains the same data as .npz file.

    Args:
        npz_path: Path to original .npz file
        h5_path: Path to converted .h5 file

    Returns:
        bool: True if data matches, False otherwise
    """
    try:
        npz_data = np.load(npz_path)

        with h5py.File(h5_path, 'r') as h5_data:
            # Check that all keys exist
            if set(npz_data.files) != set(h5_data.keys()):
                print(f"Key mismatch: npz keys={npz_data.files}, h5 keys={list(h5_data.keys())}")
                return False

            # Check that data matches
            for key in npz_data.files:
                if not np.allclose(npz_data[key], h5_data[key][:]):
                    print(f"Data mismatch for key '{key}'")
                    return False

        return True

    except Exception as e:
        print(f"Verification error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert .npz files to .h5 format")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npz files to convert')
    parser.add_argument('--delete_npz', action='store_true',
                        help='Delete original .npz files after successful conversion')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--compression_level', type=int, default=4,
                        help='gzip compression level 0-9 (default: 4, higher=better compression but slower)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify a sample of conversions before proceeding')

    args = parser.parse_args()

    # Validate directory
    if not osp.exists(args.data_dir):
        print(f"Error: Directory does not exist: {args.data_dir}")
        return

    # Optional: Verify conversion on a sample file first
    if args.verify:
        npz_files = sorted(glob(osp.join(args.data_dir, '*.npz')))
        if len(npz_files) > 0:
            print("\nVerifying conversion on first file...")
            test_npz = npz_files[0]
            test_h5 = test_npz.replace('.npz', '.h5')

            convert_single_file(test_npz, delete_npz=False, compression_level=args.compression_level)

            if verify_conversion(test_npz, test_h5):
                print("✓ Verification successful!")
                os.remove(test_h5)
            else:
                print("✗ Verification failed! Aborting.")
                if osp.exists(test_h5):
                    os.remove(test_h5)
                return

    # Proceed with conversion
    convert_directory(
        args.data_dir,
        delete_npz=args.delete_npz,
        num_workers=args.num_workers,
        compression_level=args.compression_level
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
