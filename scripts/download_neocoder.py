#!/usr/bin/env python3
"""
Download and prepare the NeoCoder dataset.

This script:
1. Clones the NeoCoder repository
2. Verifies the dataset files
3. Runs basic sanity checks

Usage:
    python scripts/download_neocoder.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Download and verify NeoCoder dataset."""
    print("="*80)
    print("NeoCoder Dataset Setup")
    print("="*80)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    neocoder_dir = data_dir / "neocoder_repo"

    print(f"\nProject root: {project_root}")
    print(f"Data directory: {data_dir}")

    # Check if already downloaded
    if neocoder_dir.exists():
        print(f"\nNeoCoder repository already exists at: {neocoder_dir}")
        print("\nTo re-download, delete the directory first:")
        print(f"  rm -rf {neocoder_dir}")
        response = input("\nContinue anyway and verify dataset? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    else:
        # Clone repository
        print("\n📥 Cloning NeoCoder repository...")
        print("This may take 1-2 minutes...")

        try:
            subprocess.run(
                ["git", "clone", "https://github.com/JHU-CLSP/NeoCoder.git", str(neocoder_dir)],
                cwd=data_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print("Repository cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")
            return 1
        except FileNotFoundError:
            print("Error: git not found. Please install git first.")
            return 1

    # Verify dataset files
    print("\nVerifying dataset files...")

    required_files = [
        "datasets/CodeForce/NeoCoder/NeoCoder.json",
        "datasets/CodeForce/NeoCoder/test_cases_annotated.json",
        "datasets/CodeForce/NeoCoder/human_solutions.json",
        "datasets/CodeForce/NeoCoder/human_solution_techniques.json",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = neocoder_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"{file_path} ({size_mb:.1f} MB)")
        else:
            print(f"{file_path} (missing)")
            all_exist = False

    if not all_exist:
        print("\nSome required files are missing!")
        return 1

    # Run sanity checks
    print("\nRunning sanity checks...")

    try:
        import json

        # Check NeoCoder.json
        neocoder_file = neocoder_dir / "datasets/CodeForce/NeoCoder/NeoCoder.json"
        with open(neocoder_file) as f:
            data = json.load(f)

        print(f"NeoCoder.json: {len(data)} problems")

        # Check test cases
        test_cases_file = neocoder_dir / "datasets/CodeForce/NeoCoder/test_cases_annotated.json"
        with open(test_cases_file) as f:
            test_cases = json.load(f)

        print(f"test_cases_annotated.json: {len(test_cases)} test case entries")

        # Verify data structure
        if len(data) > 0:
            first_problem = data[0]
            assert 'problem_id' in first_problem
            assert 'problem_statements' in first_problem
            assert 'constraints_list' in first_problem
            print(f"Data structure valid")

    except Exception as e:
        print(f"Sanity check failed: {e}")
        return 1

    # Success!
    print("\n" + "="*80)
    print("NeoCoder dataset is ready!")
    print("="*80)

    print("\nNext steps:")
    print("  1. Test the data loader:")
    print("     python src/data/neocoder_loader.py")
    print("\n  2. Test the full pipeline:")
    print("     python scripts/test_data_pipeline.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
