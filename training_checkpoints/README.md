# Training checkpoints

Weights are produced locally by `phases/Train_v2/scripts/train.py` and are **not** committed.

After a successful training run this folder typically contains:

| File | Description |
|------|-------------|
| `cGAN_generator.pth` | Generator state dict |
| `normalization_stats.json` | Fitted `ImpactTabularScaler` statistics |
| `cGAN_generator_config.json` | Training hyperparameters snapshot |

Point inference / evaluation scripts at a checkpoint with `--model_path`, for example:

```bash
python phases/Train_v2/scripts/generate.py --model_path training_checkpoints/cGAN_generator.pth \
  --velocity 120 --radius 5 --impact_x 0 --impact_y 0
```

Do not commit `*.pth` / `*.pt` files.
