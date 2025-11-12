from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Set

import pandas as pd


def _write_filtered_metadata(src: Path, dst: Path, keep_ids: Set[int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("concept_id", -1)) in keep_ids:
                fout.write(json.dumps(record) + "\n")


def _filter_parquet(src: Path, dst: Path, keep_ids: Set[int], *, keep_negative: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    mask = df["concept_id"].isin(keep_ids)
    if keep_negative and (df["concept_id"] == -1).any():
        mask |= df["concept_id"] == -1
    df.loc[mask].to_parquet(dst, index=False)


def _mirror_misc_files(src_dir: Path, dst_dir: Path, handled: Iterable[str]) -> None:
    for item in src_dir.iterdir():
        if item.name in handled:
            continue
        if item.is_file():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_dir / item.name)


def subset_single_directory(src_base: Path, dst_base: Path, keep_ids: Set[int], *, keep_negative: bool) -> None:
    generate_src = src_base / "generate"
    if generate_src.exists():
        generate_dst = dst_base / "generate"
        handled = set()
        metadata_src = generate_src / "metadata.jsonl"
        if metadata_src.exists():
            _write_filtered_metadata(metadata_src, generate_dst / "metadata.jsonl", keep_ids)
            handled.add("metadata.jsonl")
        parquet_src = generate_src / "train_data.parquet"
        if parquet_src.exists():
            _filter_parquet(parquet_src, generate_dst / "train_data.parquet", keep_ids, keep_negative=keep_negative)
            handled.add("train_data.parquet")
        _mirror_misc_files(generate_src, generate_dst, handled)

    inference_src = src_base / "inference"
    if inference_src.exists():
        inference_dst = dst_base / "inference"
        handled = set()
        latent_src = inference_src / "latent_eval_data.parquet"
        if latent_src.exists():
            _filter_parquet(latent_src, inference_dst / "latent_eval_data.parquet", keep_ids, keep_negative=False)
            handled.add("latent_eval_data.parquet")
        _mirror_misc_files(inference_src, inference_dst, handled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a concept subset from existing AxBench data.")
    parser.add_argument("--source-root", type=Path, default=Path("axbench/axbench/concept500"),
                        help="Root directory containing the original concept data (default: axbench/axbench/concept500).")
    parser.add_argument("--target-root", type=Path, required=True,
                        help="Destination root where the subset will be written.")
    parser.add_argument("--subdirs", nargs="+", required=True,
                        help="List of sub-directories to process (e.g. prod_2b_l20_v1 prod_2b_l10_v1).")
    parser.add_argument("--concept-ids", type=int, nargs="+", required=True,
                        help="Concept ids to keep (e.g. 0 1 2 3 4 5 6 7 8 9).")
    parser.add_argument("--drop-negative", action="store_true",
                        help="Remove concept_id == -1 rows from the training parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ids = set(args.concept_ids)
    keep_negative = not args.drop_negative

    for subdir in args.subdirs:
        src_base = args.source_root / subdir
        if not src_base.exists():
            raise FileNotFoundError(f"Source directory not found: {src_base}")
        dst_base = args.target_root / subdir
        subset_single_directory(src_base, dst_base, keep_ids, keep_negative=keep_negative)
        print(f"Wrote subset for {subdir} -> {dst_base}")


if __name__ == "__main__":
    main()
