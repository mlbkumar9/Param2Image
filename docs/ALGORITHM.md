# Algorithm: Param2Image cGAN

Detailed architecture, losses, scaling, and training notes for the Param2Image conditional GAN (Abaqus impact surrogate). For a shorter end-to-end method sketch, see the Method overview in the [README](../README.md).

## 1. Problem formulation

Given impact parameters

\[
\mathbf{p} = (v, r, x, y)^\top
\in \mathbb{R}^{4}
\]

where \(v\) is projectile velocity (m/s), \(r\) projectile radius (mm), and \((x,y)\) impact location (mm), the model predicts a multi-modal response

\[
\big(\hat{\mathbf{I}}, \hat{\mathbf{y}}\big)
= G(\mathbf{z}, \mathbf{p}),
\qquad
\hat{\mathbf{I}} \in [-1,1]^{4\times 512\times 512},
\quad
\hat{\mathbf{y}} \in \mathbb{R}^{3}.
\]

Here \(\mathbf{z}\sim\mathcal{N}(0,I_{100})\) is latent noise, \(\hat{\mathbf{I}}\) is an RGBA after-impact image, and

\[
\hat{\mathbf{y}} = (\hat{\sigma}_{\mathrm{Mises}}, \hat{\varepsilon}_{\mathrm{PEEQ}}, \hat{\delta})
\]

approximates peak Mises stress (MPa), peak equivalent plastic strain, and peak deflection (mm).

The discriminator \(D(\mathbf{I}, \mathbf{p})\) scores whether an image is a realistic outcome for parameters \(\mathbf{p}\).

## 2. Tabular preprocessing

Images are resized to \(512\times 512\), converted to tensors in \([0,1]\), then normalized to \([-1,1]\) with mean/std \(0.5\) per channel.

Tabular fields use `ImpactTabularScaler`:

**Parameters (standardization)**

\[
\tilde{p}_j = \frac{p_j - \mu_{p,j}}{\sigma_{p,j}}, \qquad j=1,\ldots,4.
\]

**Responses (log1p then standardize)**

\[
u_k = \log(1 + \max(y_k, 0)),
\qquad
\tilde{y}_k = \frac{u_k - \mu_{u,k}}{\sigma_{u,k}}, \qquad k=1,2,3.
\]

At inference, numerical heads emit \(\tilde{\mathbf{y}}\); the inverse map is

\[
u_k = \tilde{y}_k\,\sigma_{u,k} + \mu_{u,k},
\qquad
\hat{y}_k = \max\big(e^{u_k}-1,\,0\big)
\]

(with \(u_k\) clipped for numerical safety). Statistics are written to `normalization_stats.json` beside the checkpoint.

## 3. Generator architecture

### 3.1 Shared input

\[
\mathbf{h}_0 = [\mathbf{z};\,\tilde{\mathbf{p}}] \in \mathbb{R}^{104}.
\]

### 3.2 Image head (64 → 512 upsampling)

1. Linear projection to a \(256\times 64\times 64\) feature volume:
   \[
   \mathrm{Linear}(104 \rightarrow 256\cdot 64^{2}),\quad
   \mathrm{reshape}\rightarrow (256,64,64).
   \]
2. BatchNorm2d → Upsample(\(\times 2\)) → Conv\(3\times3\)\(256\rightarrow256\) → BN → LeakyReLU(0.2)  (64→128)
3. Upsample(\(\times 2\)) → Conv\(3\times3\)\(256\rightarrow128\) → BN → LeakyReLU  (128→256)
4. Upsample(\(\times 2\)) → Conv\(3\times3\)\(128\rightarrow64\) → BN → LeakyReLU  (256→512)
5. Conv\(3\times3\)\(64\rightarrow 4\) → Tanh  → RGBA in \([-1,1]\)

### 3.3 Numerical head modes

**Physics mode (default for new training):** deterministic in \(\tilde{\mathbf{p}}\) only (aligned with FEA response quantities):

\[
\tilde{\mathbf{p}}
\xrightarrow{\mathrm{Linear}+LN+\mathrm{SiLU}}
128
\xrightarrow{\mathrm{Linear}+LN+\mathrm{SiLU}}
128
\xrightarrow{\mathrm{Linear}+\mathrm{SiLU}}
64
\xrightarrow{\mathrm{Linear}}
3.
\]

**Legacy mode:** consumes \([\mathbf{z};\,\tilde{\mathbf{p}}]\) through a smaller MLP (`104→64→32→3` with ReLU). Retained for older checkpoints; loaders auto-detect mode from `numerical_head.0.weight` shape.

## 4. Discriminator architecture

1. Embed \(\tilde{\mathbf{p}}\) to an \(8\times 8\) map, bilinearly upsample to \(512\times 512\), concatenate with the image → \(5\) channels.
2. Five stride-2 conv blocks (with Dropout2d 0.25; BN after the first):
   \[
   512\to256\to128\to64\to32\to16
   \]
   channel widths \(5\rightarrow32\rightarrow64\rightarrow128\rightarrow256\rightarrow256\).
3. Flatten \(256\cdot16^{2}\) → Linear → scalar validity logit (LSGAN-style; no sigmoid in the network).

## 5. Losses

Let \(\mathbf{I}\) be the real after-image, \(G_I\) / \(G_y\) the generator outputs, and \(\mathbf{1}/\mathbf{0}\) adversarial targets.

**Adversarial (MSE / LSGAN)**

\[
\mathcal{L}_{D}
=
\tfrac{1}{2}\Big(
\mathrm{MSE}\big(D(\mathbf{I},\tilde{\mathbf{p}}),\,\mathbf{1}\big)
+
\mathrm{MSE}\big(D(G_I,\tilde{\mathbf{p}}),\,\mathbf{0}\big)
\Big),
\]

\[
\mathcal{L}_{G}^{\mathrm{adv}}
=
\mathrm{MSE}\big(D(G_I,\tilde{\mathbf{p}}),\,\mathbf{1}\big).
\]

**Image reconstruction**

\[
\mathcal{L}_{G}^{L1} = \|\ G_I - \mathbf{I}\ \|_{1}.
\]

**Numerical regression** (on scaled targets): Smooth L1 with \(\beta=0.5\):

\[
\mathcal{L}_{G}^{\mathrm{num}} = \mathrm{SmoothL1}_{\beta=0.5}(G_y,\,\tilde{\mathbf{y}}).
\]

**Total generator objective**

\[
\mathcal{L}_{G}
=
\mathcal{L}_{G}^{\mathrm{adv}}
+ \lambda_{L1}\,\mathcal{L}_{G}^{L1}
+ \lambda_{\mathrm{num}}\,\mathcal{L}_{G}^{\mathrm{num}},
\]

with defaults \(\lambda_{L1}=100\), \(\lambda_{\mathrm{num}}=10\).

Gradients for \(G\) and \(D\) are clipped at global norm \(5.0\).

## 6. Training procedure

| Hyperparameter | Default |
|----------------|---------|
| Epochs | 150 |
| Batch size | 2 (512×512 RGBA) |
| Optimizer | Adam, \(\mathrm{lr}=2\cdot10^{-4}\), \(\beta_1=0.5\), \(\beta_2=0.999\) |
| Latent dim | 100 |
| Numerical head | `physics` |
| Tabular scaling | on (`log1p`) |
| Device | CUDA → XPU → HIP → MPS → CPU |

Each iteration: (1) sample \(\mathbf{z}\), forward \(G\); (2) update \(G\) on \(\mathcal{L}_G\); (3) update \(D\) on \(\mathcal{L}_D\). Checkpoints every 20 epochs and at the end; TensorBoard scalars under `logs/`.

## 7. Inference

1. Load `cGAN_generator.pth` (auto-detect numerical head mode).
2. Load `normalization_stats.json` if present.
3. Standardize user parameters; sample \(\mathbf{z}\).
4. Forward \(G\); denormalize image from \([-1,1]\) to \([0,1]\) for saving; inverse-transform numerical outputs to physical units.

For legacy stochastic numerical heads, comparison scripts may average several latent draws.

## 8. Design rationale (brief)

- **Conditional image path** captures spatial impact patterns that pure tabular regressors miss.
- **Auxiliary numerical head** ties the generator to scalar FEA KPIs used in engineering reports.
- **Physics numerical head** removes latent noise from scalars that should be deterministic given \(\mathbf{p}\).
- **log1p scaling** stabilizes learning across multiplicative ranges of stress / strain / deflection.
- **LSGAN + strong L1** improves stability and pixel fidelity for a modest simulation corpus.

See [WORKFLOWS.md](WORKFLOWS.md) for end-to-end CLI usage.
