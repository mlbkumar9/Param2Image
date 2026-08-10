# Param2Image v3 — training & study plan

This document is the execution plan for re-running the Param2Image experiments
at 1024×1024 with the new training stack, and for the four ablation studies that
will populate the new "Results" section of `Pres_v2/DML_Mahendra/main.tex`.

It is paired with the code in this folder and Appendix A.2 of the paper. The
goal is to commit to one plan now so the training campaign runs unattended and
the post-processing produces the exact figures the paper expects.

---

## 1. Order of operations

**Phase A — sanity check (≈30 min on Colab T4)**

1. Open `notebook_train_p2i_colab.ipynb`.
2. Run cells 1–6. Cell 6 runs the model smoke test (`model_p2i_pytorch.py`).
   Confirm: G ≈ 80M params, D ≈ 3M params, forward pass at 1024² uses < 14 GB.
3. If VRAM > 14 GB, halve `P2I_INIT_CHANNELS` (512 → 256) and re-run cell 6
   before launching cell 7. Do **not** raise `P2I_BATCH_SIZE` above 1 on T4.

**Phase B — main training (200 epochs, ~15–18 Colab sessions)**

4. Run cell 7. The script saves `ckpt_latest.pt` to local SSD and mirrors it to
   Drive at the end of every epoch.
5. When the Colab session disconnects, re-open the notebook, run cells 1–6
   again (no change to config), then run cell 7. The script auto-resumes.
6. Keep an eye on `Val/SSIM` in TensorBoard (cell 8) every few sessions.
   Expected trajectory: SSIM crosses 0.80 around epoch 30, plateaus near 0.90
   by epoch 120, with marginal gains beyond.

**Phase C — ablations (after Phase B converges)**

7. Run the four sub-experiments described in §3.
8. Run the post-processing scripts described in §4 to generate figures.
9. Replace the placeholder numbers and figures in `main.tex` §5.

---

## 2. Why this exact recipe

Each design choice in the new stack is in `main.tex` §3.3, but for execution
this is the short version:

| Knob | Old (v2) | New (v3) | Reason |
|---|---|---|---|
| Resolution | 512² | 1024² (T4-fit) | User requirement; reveals fine fracture detail. |
| Optimizer | Adam (β₁=0.5) | AdamW (β₁=0) + weight decay | Decoupled WD, β₁=0 is the modern GAN default and reduces oscillation. |
| LR schedule | StepLR | Cosine + 5-epoch linear warmup | Smoother decay, no abrupt step that destabilizes GAN balance. |
| Adversarial loss | LSGAN, global D | LSGAN, **SN-PatchGAN D** | Local patch decision improves stress-texture fidelity. |
| G stability | none | **EMA** decay 0.999 | Standard modern-GAN trick; SSIM lift of 0.02–0.05. |
| D stability | grad clip only | **R1** every 16 steps | Lazy R1 regularization, fraction of cost of full R1. |
| Memory | none | **AMP + grad ckpt + grad-accum=4** | Fits 1024² in 16 GB. |
| Generator block | plain conv | **ResBlock** | Paper claimed this; v3 code now matches. |

All of these are individually small but compose into a meaningful change. The
**optimizer ablation** (study 2 below) measures the joint effect.

---

## 3. Ablation studies

### Study 1 — Resolution ablation

**Goal:** quantify the benefit of moving from 256² → 512² → 1024².

**Procedure**
- Train three models with identical hyperparameters except `P2I_IMG_SIZE` and
  `P2I_INIT_SIZE` (use 16/16/16 for init_size to keep init_channels at 512):
  * `IMG_SIZE=256`, `INIT_SIZE=16`, `INIT_CHANNELS=512` → 4 upsamples
  * `IMG_SIZE=512`, `INIT_SIZE=16`, `INIT_CHANNELS=512` → 5 upsamples
  * `IMG_SIZE=1024`, `INIT_SIZE=16`, `INIT_CHANNELS=512` → 6 upsamples (main run)
- Train 100 epochs each on the same 500-sample subset and the same val split.
- Evaluate all three on the same 50-sample held-out validation set.

**Metrics to report**
- Mean / std SSIM, PSNR, FID.
- Mean / std L1.
- Per-output R² (mises, peeq, deflection).
- Inference time per image on the same T4.

**Figure to add to paper** — Section 5 (new subsection "Resolution ablation"):
- `figures/v3_resolution_ablation.png`: bar chart of SSIM/PSNR/FID/R² vs resolution.

### Study 2 — Optimizer / training-stack ablation

**Goal:** isolate the contribution of the new training recipe.

**Procedure**
- Two runs at 1024² with the same architecture:
  * **Legacy:** Adam(β₁=0.5), StepLR, no EMA, no R1, no grad checkpoint, no AMP.
    (Run `P2I_EMA_DECAY=0`, `P2I_R1_GAMMA=0`, `P2I_USE_AMP=0`,
    `P2I_USE_GRAD_CHECKPOINT=0`, and override LR to constant.)
  * **New:** the default config in the notebook.
- 100 epochs each.

**Metrics**
- SSIM / PSNR / FID at epochs 20, 50, 100 (convergence speed).
- R² at epoch 100.

**Figure** — Section 5 (new subsection "Optimization ablation"):
- `figures/v3_optimizer_ablation.png`: two-panel chart, left = SSIM trajectory
  vs epoch, right = bar chart of final R² per output.

### Study 3 — Damage-regime breakdown

**Goal:** show where the model is strong and where it struggles.

**Procedure**
- For every test sample, classify into a regime:
  * `dent`     — `max_peeq < 0.05` (no element deletion)
  * `penetration` — `max_peeq ≥ 0.05` and depth < plate thickness
  * `perforation` — `max_peeq ≥ 0.05` and complete through-hole
  Use the ABAQUS log to decide the perforation flag.
- Group SSIM / PSNR / R² by regime and bootstrap 95% CIs.

**Figures** — extends Section 5.4 (Detailed analysis):
- `figures/v3_regime_metrics.png`: grouped bar chart of mean SSIM/PSNR per regime.
- `figures/v3_regime_qualitative.png`: 3×3 grid of best/median/worst predictions
  per regime (ground truth, generated, error map).

### Study 4 — Numerical-head improvement study

**Goal:** push R² for max stress and PEEQ above zero (currently negative in v2).

**Procedure**
- Sweep `P2I_LAMBDA_NUM ∈ {1, 10, 50, 100}` at fixed 512² (cheaper than 1024²)
  with the new training stack.
- Optional follow-up: replace `SmoothL1Loss` in the numerical head with a
  heteroscedastic Gaussian NLL (model predicts both µ and log σ, loss is
  `0.5*((y-µ)/σ)² + log σ`). One extra training run with the best λ_n from the
  sweep.

**Figures** — extends Section 5.4:
- `figures/v3_lambda_num_sweep.png`: R² vs λ_n for each output.
- `figures/v3_predicted_vs_actual_best.png`: predicted-vs-actual scatter at the
  best (λ_n, loss) combination, replacing the current `train_v2_predicted_vs_actual.png`.

---

## 4. Post-processing scripts to add

To keep figure generation reproducible, add the following to this folder
**after** the main run completes:

| Script | Purpose | Output |
|---|---|---|
| `evaluate_p2i_pytorch.py` (carry over from Train_v2 with v3 model loader) | Run the trained EMA model on the held-out set; produce per-sample CSV of (params, SSIM, PSNR, L1, predicted outputs, ground-truth outputs). | `Evaluation_Results/test_results.csv` |
| `make_resolution_figure.py` | Read three eval CSVs (256/512/1024) and plot the bar chart. | `figures/v3_resolution_ablation.png` |
| `make_optimizer_figure.py` | Read two eval CSVs (legacy/new). | `figures/v3_optimizer_ablation.png` |
| `make_regime_figure.py` | Read one eval CSV + ABAQUS logs; stratify; emit two charts. | `figures/v3_regime_metrics.png`, `figures/v3_regime_qualitative.png` |
| `make_lambda_sweep_figure.py` | Read four eval CSVs from the λ_n sweep. | `figures/v3_lambda_num_sweep.png` |
| `make_predicted_vs_actual.py` | One eval CSV. | `figures/v3_predicted_vs_actual_best.png` |

All of these are <100 lines of Python each (pandas + matplotlib). They should
read CSVs and not depend on the trained model directly — that way they can be
re-run on a laptop while the next training session is on Colab.

---

## 5. Where each new figure lands in `main.tex`

| Figure | Section | Notes |
|---|---|---|
| `figures/v3_resolution_ablation.png` | New §5.6 "Resolution ablation" | Add `\section*` or `\subsection`. |
| `figures/v3_optimizer_ablation.png` | New §5.7 "Optimization ablation" | Cite both Adam and AdamW \cite{loshchilov2019adamw}. |
| `figures/v3_regime_metrics.png` | §5.4 (extends) | After Fig. `corr_matrix`. |
| `figures/v3_regime_qualitative.png` | §5.3 (extends) | After Fig. `worst_samples`. |
| `figures/v3_lambda_num_sweep.png` | §5.4 (extends) | After Fig. `pred_vs_actual`. |
| `figures/v3_predicted_vs_actual_best.png` | §5.4 — **replaces** `train_v2_predicted_vs_actual.png` | |

The placeholders in §5 of `main.tex` are written so this swap is mechanical:
update the `\includegraphics` paths and the SSIM / R² numbers in §5.5 ("Key
findings") with the new values.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Colab disconnects mid-epoch. | Checkpoints are saved at end of every epoch and mirrored to Drive. Auto-resume restores the optimizer, EMA, AMP scaler, scheduler, RNG state. Worst-case loss: one epoch. |
| Drive quota fills up. | Only `ckpt_latest.pt` + `ckpt_best.pt` are mirrored (each ~250–350 MB for 80 M params + FP32 opt state). Total budget ~1 GB. |
| OOM on T4 despite the T4-fit config. | First fallback: `P2I_GRAD_ACCUM=8`, `P2I_USE_GRAD_CHECKPOINT=1` (already on). Second fallback: drop `P2I_INIT_CHANNELS` 512 → 256 (cuts G parameters from 80 M to ~25 M; still fits the paper's claim of "real 1024² output"). Third fallback: temporarily drop `P2I_IMG_SIZE=768` and document the change. |
| Training diverges (D collapses or G saturates). | (a) Check the `Loss/R1` curve in TensorBoard — if it spikes >50, raise `P2I_R1_INTERVAL` (more frequent regularization). (b) If `Loss/G_adv` flatlines, lower `P2I_LR_G` to 5e-5 and resume from `ckpt_latest.pt`. |
| Stress / PEEQ R² stays negative even after Study 4. | Promote to physics-informed loss as listed in §6 (Future work) — outside the scope of the current paper revision, but the file structure already supports it. |
| Free Colab GPU quota exhausted. | Switch the runtime type back to CPU; the training will refuse to launch (it requires CUDA in `pick_device`). Wait until quota resets and continue from `ckpt_latest.pt`. |

---

## 7. Wall-clock budget

| Step | Budget | Notes |
|---|---|---|
| Phase A (sanity check) | 30 min | One Colab session. |
| Phase B (main 1024² run, 200 epochs) | ~190 h | At ~1 h/epoch on T4. Spread across 16 sessions. |
| Study 1 (256², 100 epochs) | ~25 h | Reuses the same script with `P2I_IMG_SIZE=256`. |
| Study 1 (512², 100 epochs) | ~60 h | Optional — Train_v2's existing checkpoints already give a 512² baseline; use those instead and save ~60 h. |
| Study 2 (legacy 1024², 100 epochs) | ~100 h | Slower than v3 because no AMP. |
| Study 3 | < 30 min | Pure post-processing. |
| Study 4 (4 × 50 epochs at 512²) | ~60 h | Cheaper at 512²; the conclusion transfers. |
| Total | **~7 weeks of wall-clock on free Colab**, in 12 h shards. |

**Recommendation:** do Phase B and Study 3 first (they are the headline
results), then run Study 2 and Study 4 in parallel with paper revisions, and do
Study 1 last because Train_v2 already supplies two of the three resolutions.

---

## 8. Definition of done

- `ckpt_best.pt` reaches Val/SSIM ≥ 0.88 on the 50-sample validation set.
- The five new figures in §5 of this document exist under
  `Pres_v2/DML_Mahendra/figures/`.
- `main.tex` §5 numbers (mean SSIM, PSNR, R²) are replaced with the new run's
  numbers; the placeholder caveat in §5 ("These will be updated…") is removed.
- `Pres_v2/DML_Mahendra/main.tex` builds cleanly with the new bibliography keys
  (`loshchilov2019adamw`, `arjovsky2017wgan`, `gulrajani2017wgangp`).
- Tag `v3-results-locked` in git when the above is true.
