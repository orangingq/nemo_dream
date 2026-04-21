from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified seed -> stage7 pipeline.",
    )
    parser.add_argument("--input", default="data/curated/prosocial_input.jsonl")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cultural-map", default="conf/cultural_map.json")
    parser.add_argument("--judge", choices=["mock", "nim"], default="nim")
    parser.add_argument("--chart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "generate.py",
        f"--seed_dataset={args.input}",
        f"--output_dir={args.curated_dir}",
        f"--judge={args.judge}",
        f"--chart={str(args.chart).lower()}",
        f"--visualize={str(args.visualize).lower()}",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--cultural_map={args.cultural_map}",
    ]
    if args.limit is not None:
        cmd.append(f"--limit={args.limit}")
    if args.model:
        cmd.append(f"--model={args.model}")
    if args.base_url:
        cmd.append(f"--base_url={args.base_url}")

    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
