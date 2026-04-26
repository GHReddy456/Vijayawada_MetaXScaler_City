#!/usr/bin/env python3
"""
Quantitative baseline vs improved (rule-based) comparison for judges.

GRPO-trained policy should be compared the same way once a checkpoint is loaded
in Colab or locally; this script documents the evaluation harness.

Usage (from backend/):
  python scripts/compare_baseline_vs_improved.py --seeds 20 --level 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from evaluate import evaluate_episode
from policies import baseline_policy, improved_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="Number of distinct episode seeds")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("../docs/results/baseline_vs_improved.json"),
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    base_rewards: List[float] = []
    impr_rewards: List[float] = []

    for seed in range(args.seeds):
        b = evaluate_episode(
            baseline_policy,
            level=args.level,
            seed=1000 + seed,
            max_steps=args.max_steps,
            policy_name="baseline",
        )
        i = evaluate_episode(
            improved_policy,
            level=args.level,
            seed=1000 + seed,
            max_steps=args.max_steps,
            policy_name="improved",
        )
        br, ir = float(b["total_reward"]), float(i["total_reward"])
        base_rewards.append(br)
        impr_rewards.append(ir)
        rows.append(
            {
                "seed": 1000 + seed,
                "baseline_reward": br,
                "improved_reward": ir,
                "delta_reward": ir - br,
                "baseline_deaths": int(b["deaths"]),
                "improved_deaths": int(i["deaths"]),
            }
        )

    summary = {
        "level": args.level,
        "max_steps": args.max_steps,
        "n_seeds": args.seeds,
        "baseline_reward_mean": round(statistics.mean(base_rewards), 3),
        "improved_reward_mean": round(statistics.mean(impr_rewards), 3),
        "mean_delta_reward": round(statistics.mean([r["delta_reward"] for r in rows]), 3),
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
