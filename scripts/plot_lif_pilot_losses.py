#!/usr/bin/env python3
"""Plot the immutable original-LIF pilot histories used for flight diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = (
    (
        "LR 1e-4",
        "#2563eb",
        "-",
        Path("runs/crazyflie-lif-original-pilot-500k-v1"),
    ),
    (
        "LR 3e-5",
        "#d97706",
        "--",
        Path("runs/crazyflie-lif-original-lr3e-5-100k-v1"),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    paths = sorted((run_dir / "history").glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"No history JSONL files found under {run_dir}")

    rows: list[dict] = []
    sources: list[dict] = []
    for path in paths:
        sources.append({"path": str(path), "sha256": sha256(path)})
        with path.open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())

    updates = [int(row["completed_updates"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if updates != expected:
        raise RuntimeError(f"History updates are not contiguous in {run_dir}")
    for metric in ("loss", "value_loss", "policy_loss"):
        if any(not math.isfinite(float(row[metric])) for row in rows):
            raise RuntimeError(f"Non-finite {metric} in {run_dir}")
    return rows, sources


def trailing_mean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/lif_pilot_loss_comparison.png"),
    )
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")

    loaded = []
    metadata = {
        "schema_version": 1,
        "figure": str(args.output),
        "rolling_window_updates": args.window,
        "runs": [],
    }
    for label, color, linestyle, run_dir in RUNS:
        rows, sources = load_run(run_dir)
        loaded.append((label, color, linestyle, rows))
        metadata["runs"].append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "row_count": len(rows),
                "first_update": int(rows[0]["completed_updates"]),
                "last_update": int(rows[-1]["completed_updates"]),
                "total_interactions": int(rows[-1]["total_interactions"]),
                "successful_episode_count_sum": int(
                    sum(row["successful_episode_count"] for row in rows)
                ),
                "completed_episode_count_sum": int(
                    sum(row["completed_episode_count"] for row in rows)
                ),
                "sources": sources,
            }
        )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metric_specs = (
        ("loss", "PPO total loss", "Loss"),
        ("value_loss", "Value loss", "Loss"),
        ("policy_loss", "Policy loss", "Loss"),
    )

    for axis, (metric, title, ylabel) in zip(axes, metric_specs, strict=True):
        for label, color, linestyle, rows in loaded:
            updates = [int(row["completed_updates"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            axis.plot(
                updates,
                values,
                color=color,
                linewidth=0.65,
                alpha=0.16,
            )
            axis.plot(
                updates,
                trailing_mean(values, args.window),
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=f"{label} — trailing mean ({args.window})",
            )
        axis.axhline(0.0, color="#475569", linewidth=0.7, alpha=0.7)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.55)
        axis.legend(loc="upper right", frameon=False)

    axes[-1].set_xlabel("PPO update (4 environments × 25 steps = 100 interactions/update)")
    fig.suptitle(
        "Original frozen LIF pilot training losses — 100,000 interactions",
        fontsize=15,
        fontweight="bold",
        x=0.07,
        ha="left",
    )
    fig.text(
        0.07,
        0.012,
        "Thin lines: raw update values. Thick lines: trailing mean. "
        "Both pilots recorded zero successful episodes; lower PPO loss alone does not establish flight success.",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.99, 0.955))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
