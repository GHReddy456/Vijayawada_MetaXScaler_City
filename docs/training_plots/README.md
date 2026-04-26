# Training plots

| File | Meaning |
|------|---------|
| `reward_vs_episode.png` | Episode-level reward from GRPO artifact JSON (or demo curve). |
| `loss_vs_step.png` | Optional TRL scalar loss vs step if `training_metrics.json` is present. |

**Demo vs real runs:** The checked-in PNGs may be generated from `docs/demo_grpo_artifact/` for layout review. For judging, regenerate from your real checkpoint:

```bash
cd backend
pip install -r requirements-train.txt
python scripts/plot_training_run.py --checkpoint-dir checkpoints/crisisworld-grpo --out-dir ../docs/training_plots
```
