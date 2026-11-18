"""
Generator and Discriminator for a Parameters-to-Image cGAN (PyTorch).

Conditional GAN that generates post-impact RGBA images and numerical FEA
responses from a vector of impact parameters.
"""
import torch
import torch.nn as nn


NUMERICAL_HEAD_MODES = {"legacy", "physics"}


def _load_checkpoint_state_dict(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("generator_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def infer_numerical_head_mode(state_dict, num_params):
    first_weight = state_dict.get("numerical_head.0.weight")
    if first_weight is not None and first_weight.ndim == 2 and first_weight.shape[1] == num_params:
        return "physics"
    return "legacy"


def load_generator_checkpoint(
    model_path,
    device,
    num_params,
    num_numerical_outputs,
    latent_dim=100,
    img_channels=4,
    numerical_head_mode="auto",
):
    """Load old and new generator checkpoints without breaking compatibility."""
    state_dict = _load_checkpoint_state_dict(model_path, device)
    if numerical_head_mode == "auto":
        numerical_head_mode = infer_numerical_head_mode(state_dict, num_params)

    generator = Generator(
        num_params=num_params,
        num_numerical_outputs=num_numerical_outputs,
        latent_dim=latent_dim,
        img_channels=img_channels,
        numerical_head_mode=numerical_head_mode,
    ).to(device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator, numerical_head_mode


class Generator(nn.Module):
    """
    Multi-modal generator: latent noise + parameters → image + numerical outputs.
    """

    def __init__(
        self,
        num_params,
        num_numerical_outputs,
        latent_dim=100,
        img_channels=4,
        numerical_head_mode="legacy",
    ):
        super(Generator, self).__init__()
        if numerical_head_mode not in NUMERICAL_HEAD_MODES:
            raise ValueError(f"numerical_head_mode must be one of {sorted(NUMERICAL_HEAD_MODES)}")
        self.numerical_head_mode = numerical_head_mode

        input_dim = num_params + latent_dim

        # Image head: init 64×64 → upsample to 512×512
        self.init_size = 512 // 8  # 64
        self.l1 = nn.Sequential(nn.Linear(input_dim, 256 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),  # 64 -> 128
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 128 -> 256
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 256 -> 512
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        # Numerical head: physics (params-only) or legacy (noise+params)
        if numerical_head_mode == "physics":
            self.numerical_head = nn.Sequential(
                nn.Linear(num_params, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, num_numerical_outputs),
            )
        else:
            self.numerical_head = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_numerical_outputs),
            )

    def forward(self, noise, params):
        gen_input = torch.cat((noise, params), -1)

        img_out = self.l1(gen_input)
        img_out = img_out.view(img_out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(img_out)

        numerical_input = params if self.numerical_head_mode == "physics" else gen_input
        numerical_outputs = self.numerical_head(numerical_input)

        return img, numerical_outputs


class Discriminator(nn.Module):
    """
    Conditional discriminator: scores whether an image is realistic for given parameters.
    """

    def __init__(self, num_params, img_channels=4, img_size=512):
        super(Discriminator, self).__init__()

        self.param_embed_size = 8
        self.img_size = img_size
        self.param_embedding = nn.Sequential(
            nn.Linear(num_params, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.param_embed_size * self.param_embed_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # 512 → 256 → 128 → 64 → 32 → 16
        self.model = nn.Sequential(
            *discriminator_block(img_channels + 1, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
        )

        ds_size = img_size // 2 ** 5  # 16
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1))

    def forward(self, img, params):
        batch_size = img.size(0)
        param_emb = self.param_embedding(params)
        param_emb = param_emb.view(batch_size, 1, self.param_embed_size, self.param_embed_size)
        param_emb = nn.functional.interpolate(
            param_emb, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        d_in = torch.cat((img, param_emb), dim=1)

        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def main():
    IMG_SIZE = 512
    IMG_CHANNELS = 4
    NUM_PARAMS = 4
    NUM_NUMERICAL_OUTPUTS = 3
    LATENT_DIM = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Building Generator ---")
    generator = Generator(
        num_params=NUM_PARAMS,
        latent_dim=LATENT_DIM,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        img_channels=IMG_CHANNELS,
        numerical_head_mode="physics",
    ).to(device)

    print("\n--- Building Discriminator ---")
    discriminator = Discriminator(
        num_params=NUM_PARAMS, img_size=IMG_SIZE, img_channels=IMG_CHANNELS
    ).to(device)

    print("\n--- Verifying model outputs ---")
    dummy_noise = torch.randn(4, LATENT_DIM).to(device)
    dummy_params = torch.randn(4, NUM_PARAMS).to(device)
    dummy_after_image = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

    generated_image, numerical_preds = generator(dummy_noise, dummy_params)
    print("Generated image shape:", generated_image.shape)
    print("Predicted numerical outputs shape:", numerical_preds.shape)

    disc_out = discriminator(dummy_after_image, dummy_params)
    print("Discriminator output shape:", disc_out.shape)
    print("---------------------------\n")


if __name__ == "__main__":
    main()
