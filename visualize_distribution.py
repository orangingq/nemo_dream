import json
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = ["consistency", "naturalness", "relevance", "coherence", "fluency", "final"]

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize evaluation score distribution.")
    parser.add_argument("--input", required=True, help="Path to evaluated JSONL file")
    parser.add_argument("--output", default="score_distribution.png", help="Path to save chart image")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError("No rows found in input file.")

    final_scores = [row["final"] for row in rows]
    mean_scores = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_axes([0.08, 0.15, 0.38, 0.72])
    ax1.hist(final_scores, bins=5, edgecolor="black")
    ax1.set_title("Final Score Distribution")
    ax1.set_xlabel("Final score")
    ax1.set_ylabel("Count")

    ax2 = fig.add_axes([0.58, 0.15, 0.34, 0.72])
    ax2.bar(list(mean_scores.keys()), list(mean_scores.values()), edgecolor="black")
    ax2.set_title("Average Score by Axis")
    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Average score")
    ax2.set_ylim(0, 1.0)
    for label in ax2.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    fig.suptitle("Korean Data Pipeline Evaluation Overview")
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved chart to: {args.output}")

if __name__ == "__main__":
    main()
