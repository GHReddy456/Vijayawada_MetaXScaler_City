#!/usr/bin/env python3
"""
Generate labelled PNG plots from CrisisWorld GRPO checkpoint artifacts.

Reads JSON written by train_grpo.py (_save_artifacts): reward_curve.json, etc.

Usage (from backend/):
  python scripts/plot_training_run.py --checkpoint-dir checkpoints/crisisworld-grpo
  python scripts/plot_training_run.py --checkpoint-dir checkpoints/crisisworld-grpo --out-dir ../docs/training_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _plot_xy(
    xs: List[float],
    ys: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(xs, ys, color="#2563eb", linewidth=1.5, label="run")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot CrisisWorld training artifacts")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/crisisworld-grpo"),
        help="Directory containing reward_curve.json / reward_history.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../docs/training_plots"),
        help="Output directory for PNG files",
    )
    args = p.parse_args()
    ckpt: Path = args.checkpoint_dir
    out: Path = args.out_dir

    curve = _load_json(ckpt / "reward_curve.json")
    if curve is None:
        rh = _load_json(ckpt / "reward_history.json")
        if rh and "rewards" in rh:
            ys = [float(x) for x in rh["rewards"]]
            xs = list(range(len(ys)))
            curve = {"episodes": xs, "rewards": ys}

    if curve and "rewards" in curve:
        xs = [float(x) for x in curve["episodes"]]
        ys = [float(x) for x in curve["rewards"]]
        _plot_xy(
            xs,
            ys,
            title="CrisisWorld GRPO — episode reward",
            xlabel="Episode index (trainer logging)",
            ylabel="Total reward (training objective)",
            out_path=out / "reward_vs_episode.png",
        )
        print(f"Wrote {out / 'reward_vs_episode.png'}")

    deaths = _load_json(ckpt / "deaths_curve.json")
    if deaths and "deaths" in deaths:
        xs = [float(x) for x in deaths["episodes"]]
        ys = [float(x) for x in deaths["deaths"]]
        _plot_xy(
            xs,
            ys,
            title="CrisisWorld — deaths per logged episode",
            xlabel="Episode index",
            ylabel="Deaths (count)",
            out_path=out / "deaths_vs_episode.png",
        )
        print(f"Wrote {out / 'deaths_vs_episode.png'}")

    panic = _load_json(ckpt / "panic_curve.json")
    if panic and "panic" in panic:
        xs = [float(x) for x in panic["episodes"]]
        ys = [float(x) for x in panic["panic"]]
        _plot_xy(
            xs,
            ys,
            title="CrisisWorld — panic level per logged episode",
            xlabel="Episode index",
            ylabel="Panic level",
            out_path=out / "panic_vs_episode.png",
        )
        print(f"Wrote {out / 'panic_vs_episode.png'}")

    metrics = _load_json(ckpt / "training_metrics.json")
    if metrics and "step" in metrics and "trainer_loss" in metrics:
        xs = [float(x) for x in metrics["step"]]
        ys = [float(x) for x in metrics["trainer_loss"]]
        _plot_xy(
            xs,
            ys,
            title="CrisisWorld GRPO — trainer reported loss (scalar)",
            xlabel="Training step",
            ylabel="Loss (from TRL logs)",
            out_path=out / "loss_vs_step.png",
        )
        print(f"Wrote {out / 'loss_vs_step.png'}")


if __name__ == "__main__":
    main()
