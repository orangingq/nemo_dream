from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full seed -> Stage1 -> ... -> Stage7 pipeline."
    )
    parser.add_argument("--input", default="data/curated/prosocial_input.jsonl")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalized-out", default="raw/seed_normalized.jsonl")
    parser.add_argument("--validation-out-dir", default="val_out_seed")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cultural-map", default="conf/cultural_map.json")
    parser.add_argument("--chart", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    curated_dir = Path(args.curated_dir)
    stage3 = curated_dir / "stage3.jsonl"

    build_cmd = [
        sys.executable,
        "scripts/build_curated_stages.py",
        "--input",
        args.input,
        "--out-dir",
        str(curated_dir),
        "--cultural-map",
        args.cultural_map,
    ]
    if args.limit is not None:
        build_cmd.extend(["--limit", str(args.limit)])
    if args.model:
        build_cmd.extend(["--model", args.model])
    if args.base_url:
        build_cmd.extend(["--base-url", args.base_url])
    run(build_cmd)

    validation_cmd = [
        sys.executable,
        "-m",
        "pipeline.run",
        "judge=nim",
        f"generate.source_jsonl={stage3}",
        f"generate.out_jsonl={args.normalized_out}",
        f"paths.out_dir={args.validation_out_dir}",
        f"report.chart={str(args.chart).lower()}",
    ]
    run(validation_cmd)

    print("Full pipeline complete.", flush=True)
    print(f"Curated stages: {curated_dir}", flush=True)
    print(f"Normalized validation input: {args.normalized_out}", flush=True)
    print(f"Accepted/rejected/report: {args.validation_out_dir}", flush=True)


if __name__ == "__main__":
    main()
