from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert allenai/prosocial-dialog rows into pipeline seed JSONL."
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", default="data/curated/prosocial_input.jsonl")
    parser.add_argument(
        "--text-field",
        default="context",
        choices=["context", "response", "context_response"],
        help="Which dataset field to expose as the seed text.",
    )
    parser.add_argument(
        "--safety-label",
        action="append",
        default=[],
        help="Optional safety_label filter. Repeat for multiple labels, e.g. --safety-label __casual__.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source filter. Repeat for multiple sources, e.g. --source socialchemistry.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream from Hugging Face instead of materializing the split first.",
    )
    return parser.parse_args()


def seed_text(row: dict[str, Any], text_field: str) -> str:
    context = str(row.get("context") or "").strip()
    response = str(row.get("response") or "").strip()
    if text_field == "context":
        return context
    if text_field == "response":
        return response
    if context and response:
        return f"{context}\n\n{response}"
    return context or response


def include_row(row: dict[str, Any], labels: set[str], sources: set[str]) -> bool:
    if labels and row.get("safety_label") not in labels:
        return False
    if sources and row.get("source") not in sources:
        return False
    return True


def convert_row(row: dict[str, Any], idx: int, text_field: str) -> dict[str, Any]:
    dialogue_id = row.get("dialogue_id")
    response_id = row.get("response_id")
    return {
        "id": f"prosocial-{dialogue_id}-{response_id}-{idx}",
        "text": seed_text(row, text_field),
        "metadata": {
            "dataset": "allenai/prosocial-dialog",
            "split": row.get("_split"),
            "source": row.get("source"),
            "safety_label": row.get("safety_label"),
            "dialogue_id": dialogue_id,
            "response_id": response_id,
            "episode_done": row.get("episode_done"),
            "rots": row.get("rots") or [],
            "safety_annotations": row.get("safety_annotations") or [],
            "safety_annotation_reasons": row.get("safety_annotation_reasons") or [],
            "etc": row.get("etc"),
            "text_field": text_field,
        },
    }


def iter_rows(split: str, streaming: bool):
    from datasets import load_dataset

    dataset = load_dataset("allenai/prosocial-dialog", split=split, streaming=streaming)
    for row in dataset:
        item = dict(row)
        item["_split"] = split
        yield item


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = set(args.safety_label)
    sources = set(args.source)

    written = 0
    seen = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_rows(args.split, args.streaming):
            if not include_row(row, labels, sources):
                continue
            text = seed_text(row, args.text_field)
            if not text:
                continue
            if seen < args.offset:
                seen += 1
                continue
            converted = convert_row(row, written, args.text_field)
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"Wrote {written} seed rows to {output}")
    if labels:
        print(f"Filtered safety_label: {', '.join(sorted(labels))}")
    if sources:
        print(f"Filtered source: {', '.join(sorted(sources))}")


if __name__ == "__main__":
    main()
