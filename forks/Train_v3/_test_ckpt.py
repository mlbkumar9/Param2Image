import torch
from p2i_paths import checkpoint_dir
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

ck_path = checkpoint_dir() / "ckpt_best.pt"
alt = checkpoint_dir("training_checkpoints_position") / "ckpt_best.pt"
path = ck_path if ck_path.exists() else alt
ck = torch.load(path, map_location="cpu", weights_only=False)
print("Loaded:", path)
print("Checkpoint keys:", list(ck.keys()))
print("Epoch:", ck.get("epoch"))
print("best_ssim:", ck.get("best_ssim"))
cfg = ck.get("config", {})
print("Config:", cfg)
