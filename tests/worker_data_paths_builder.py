#!/usr/bin/env python3
"""Build per-worker fixture folders using --workers; test CSVs go to the first worker."""

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def read_data_model_metadata(cdes_file):
    with open(cdes_file) as f:
        meta = json.load(f)
    return meta["code"], meta["version"]


def get_datasets_per_worker(dirpath, filenames, worker_ids):
    """Distribute CSVs to local workers (round robin)."""
    datasets = {w: [] for w in worker_ids}

    if not worker_ids:
        return datasets

    if len(worker_ids) == 1:
        w = worker_ids[0]
        datasets[w] = sorted(
            os.path.join(dirpath, f)
            for f in filenames
            if f.endswith(".csv") and not f.endswith("test.csv")
        )
        return datasets

    first = worker_ids[0]
    first_csvs = sorted(
        os.path.join(dirpath, f)
        for f in filenames
        if f.endswith("0.csv") and not f.endswith("10.csv")
    )
    datasets[first].extend(first_csvs)

    remaining = sorted(
        os.path.join(dirpath, f)
        for f in filenames
        if f.endswith(".csv") and not f.endswith("0.csv") and not f.endswith("test.csv")
    )
    cycle = itertools.cycle(worker_ids[1:])
    for path in remaining:
        datasets[next(cycle)].append(path)

    return datasets


def concatenate_csv_files(csv_paths, dest: Path):
    if not csv_paths:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(csv_paths)

    canonical_header = None
    with open(dest, "w", newline="") as out:
        writer = None
        for path in csv_paths:
            with open(path, newline="") as src:
                reader = csv.DictReader(src)

                if reader.fieldnames is None:
                    continue

                if canonical_header is None:
                    canonical_header = reader.fieldnames
                    writer = csv.DictWriter(out, fieldnames=canonical_header)
                    writer.writeheader()
                else:
                    if set(reader.fieldnames) != set(canonical_header):
                        raise ValueError("Inconsistent CSV headers detected.")

                for row in reader:
                    writer.writerow(
                        {field: row.get(field, "") for field in canonical_header}
                    )


def format_directory_structure(root: Path) -> str:
    entries = sorted(e for e in os.listdir(root) if (root / e).is_dir())
    lines = [f"{root.name}/"]
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(f"{connector}{entry}")
    return "\n".join(lines)


def directory_size_bytes(root: Path) -> int:
    total = 0
    for entry in root.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def format_bytes(total: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(total)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def create_combined_datasets(structure, worker_ids, test_data_folder: Path):
    combined_root = test_data_folder / ".data_paths"
    if combined_root.exists():
        shutil.rmtree(combined_root)
    combined_root.mkdir()

    for worker_id in worker_ids:
        w_dir = combined_root / worker_id
        w_dir.mkdir()

        for (code, version), entry in sorted(structure.get(worker_id, {}).items()):
            datasets = entry.get("datasets", [])
            metadata_path = entry.get("metadata")

            if not datasets or not metadata_path:
                continue

            model_dir = w_dir / f"{code}_{version}"
            model_dir.mkdir()

            shutil.copyfile(metadata_path, model_dir / "CDEsMetadata.json")

            concatenate_csv_files(
                datasets,
                model_dir / f"{code}.csv",
            )

    return combined_root


def structure_data(test_data_folder: Path, workers):
    """Main logic: scans datasets, distributes CSVs, builds folders."""

    worker_dataset_structure = defaultdict(dict)
    all_workers = list(workers)

    for dirpath, dirnames, filenames in os.walk(test_data_folder):
        if ".data_paths" in dirnames:
            dirnames.remove(".data_paths")
        if "CDEsMetadata.json" not in filenames:
            continue

        cdes = os.path.join(dirpath, "CDEsMetadata.json")
        code, version = read_data_model_metadata(cdes)
        key = (code, version)

        local_dist = get_datasets_per_worker(dirpath, filenames, workers)

        # Test fixtures go to the first worker.
        test_csvs = [
            os.path.join(dirpath, f) for f in filenames if f.endswith("test.csv")
        ]
        local_dist[workers[0]].extend(test_csvs)

        full = local_dist

        for worker_id, csv_paths in full.items():
            if not csv_paths:
                continue

            entry = worker_dataset_structure[worker_id].setdefault(
                key, {"metadata": cdes, "datasets": []}
            )
            entry["datasets"].extend(csv_paths)

    return create_combined_datasets(
        worker_dataset_structure, all_workers, test_data_folder
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Structure and combine test datasets per worker."
    )
    p.add_argument(
        "--test-data-folder",
        type=Path,
        default=Path("tests/test_data"),
        help="Folder containing the test datasets (default: tests/test_data)",
    )
    p.add_argument(
        "--workers",
        nargs="+",
        required=True,
        help="Identifiers of workers (required)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.workers:
        print(
            "ERROR: --workers must be provided with at least one worker.",
            file=sys.stderr,
        )
        sys.exit(1)

    combined_root = structure_data(
        test_data_folder=args.test_data_folder,
        workers=args.workers,
    )

    print(f"Data paths for each worker is now located:")
    print(f"{args.test_data_folder}/{format_directory_structure(combined_root)}")
    total_size = directory_size_bytes(combined_root)
    print(f"Total structured data size: {format_bytes(total_size)}")


if __name__ == "__main__":
    main()
