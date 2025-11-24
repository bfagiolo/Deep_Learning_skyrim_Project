import os
import math
import glob
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchgen.api.cpp import return_names
from torchvision import transforms
from torchvision.utils import save_image

import time

# ============================================================
# 1. CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXTURE_DIR = "png_input/normals"      # folder of high-res textures
SAVE_DIR = "runs_half_diff"

MODEL_DIR        = os.path.join(SAVE_DIR, "model")
TRAIN_SAMPLES_DIR = os.path.join(SAVE_DIR, "train_samples")
NOISE_TESTS_DIR   = os.path.join(SAVE_DIR, "noise_tests")
RESULTS_DIR       = os.path.join(SAVE_DIR, "results")

# create folders
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN_SAMPLES_DIR, exist_ok=True)
os.makedirs(NOISE_TESTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


DEBUG_FAST_RUN = False

if DEBUG_FAST_RUN:
    EPOCHS      = 5
    HR_SIZE     = 128   # resize textures to this square
    TIMESTEPS   = 200   # total diffusion steps
    BATCH_SIZE  = 4
    CHANNELS    = 32
    MAX_IMAGES  = 200   # optionally limit dataset size
else:
    EPOCHS      = 40
    HR_SIZE     = 256
    TIMESTEPS   = 500
    BATCH_SIZE  = 4
    CHANNELS    = 48
    MAX_IMAGES  = None

LR = 2e-4

# ============================================================
# 2. DATASET
# ============================================================

class TextureDataset(Dataset):
    def __init__(self, root_dir, img_size=128, max_images=None):
        self.paths = sorted(
            glob.glob(os.path.join(root_dir, "*"))
        )
        if max_images is not None:
            self.paths = self.paths[:max_images]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                      # [0,1]
            transforms.Normalize(0.5, 0.5)             # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img

# ============================================================
# 3. NOISE SCHEDULE + DIFFUSION UTILITIES
# ============================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_hat = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

# Precompute helpful terms
sqrt_alpha_hat      = torch.sqrt(alpha_hat)
sqrt_one_minus_ahat = torch.sqrt(1.0 - alpha_hat)
sqrt_recip_alphas   = torch.sqrt(1.0 / alphas)
one_minus_alpha     = 1.0 - alphas

# For posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas[1:] * (1.0 - alpha_hat[:-1]) / (1.0 - alpha_hat[1:])


def extract(a, t, x_shape):
    """
    Helper to gather coefficients at given timesteps t
    and reshape them for broadcasting.
    a: [T]
    t: [B] (Long)
    returns: [B, 1, 1, 1]
    """
    b = a.gather(-1, t)
    return b.reshape(-1, 1, 1, 1)


def q_sample(x0, t, noise=None):
    """
    Forward diffusion: q(x_t | x_0)
    x0: [B, C, H, W]
    t: [B] Long, timestep
    """
    if noise is None:
        noise = torch.randn_like(x0)
    a_hat = extract(sqrt_alpha_hat, t, x0.shape)
    om_a = extract(sqrt_one_minus_ahat, t, x0.shape)
    return a_hat * x0 + om_a * noise


def add_noise_k(x0, k):
    """
    ADD K STEPS OF NOISE to simulate GAN output.
    k is an integer 0 <= k < TIMESTEPS.
    """
    if k < 0 or k >= TIMESTEPS:
        raise ValueError(f"k must be in [0, {TIMESTEPS-1}], got {k}")
    bsz = x0.shape[0]
    t = torch.full((bsz,), k, device=x0.device, dtype=torch.long)
    x_k = q_sample(x0, t)
    return x_k

# ============================================================
# 4. TIME EMBEDDING + SIMPLE UNET
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] (Long or float)
        output: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.linspace(0, 1, half, device=t.device)
        )  # [half]
        # turn t into float
        t = t.float().unsqueeze(1)  # [B,1]
        args = t * freqs  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


def conv_block(in_ch, out_ch, time_emb_dim=None):
    """
    Conv block with optional time embedding.
    """
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.act = nn.SiLU()
            if time_emb_dim is not None:
                self.time_mlp = nn.Linear(time_emb_dim, out_ch)
            else:
                self.time_mlp = None

        def forward(self, x, t_emb=None):
            h = self.conv1(x)
            if self.time_mlp is not None and t_emb is not None:
                # t_emb: [B, time_emb_dim] -> [B, out_ch, 1,1]
                temb = self.time_mlp(t_emb)
                h = h + temb[:, :, None, None]
            h = self.act(h)
            h = self.conv2(h)
            h = self.act(h)
            return h
    return Block()


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, time_emb_dim=128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        self.down1 = conv_block(in_ch, base_ch, time_emb_dim)
        self.down2 = conv_block(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = conv_block(base_ch * 2, base_ch * 4, time_emb_dim)

        self.pool = nn.AvgPool2d(2)

        self.mid = conv_block(base_ch * 4, base_ch * 4, time_emb_dim)

        self.up3 = conv_block(base_ch * 4 + base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2 = conv_block(base_ch * 2 + base_ch * 2, base_ch, time_emb_dim)
        self.up1 = conv_block(base_ch + base_ch, base_ch, time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        # time embedding
        t_emb = self.time_emb(t)

        # down
        h1 = self.down1(x, t_emb)          # [B, C, H, W]
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)

        # mid
        h_mid = self.mid(self.pool(h3), t_emb)

        # up
        u3 = self.upsample(h_mid)
        u3 = torch.cat([u3, h3], dim=1)
        u3 = self.up3(u3, t_emb)

        u2 = self.upsample(u3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = self.upsample(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1(u1, t_emb)

        out = self.final(u1)
        return out  # predicts noise epsilon

# ============================================================
# 5. TRAINING LOOP
# ============================================================

def p_losses(model, x0, t):
    """
    Standard DDPM loss: predict noise added at time t.
    """
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)


def train():
    start_time = time.time()
    dataset = TextureDataset(TEXTURE_DIR, img_size=HR_SIZE, max_images=MAX_IMAGES)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    model = SimpleUNet(in_ch=3, base_ch=CHANNELS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    global_step = 0

    for epoch in range(EPOCHS):
        for x0 in dl:
            x0 = x0.to(DEVICE)  # [-1,1]

            # sample random timesteps in [0, T-1]
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=DEVICE).long()

            loss = p_losses(model, x0, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {loss.item():.4f}")

        # quick sample from pure noise just to see if it's learning
        with torch.no_grad():
            x_sample = sample_from_noise(model, n=4)
            save_image(
                (x_sample + 1) / 2,
                os.path.join(TRAIN_SAMPLES_DIR, f"sample_epoch_{epoch + 1}.png"),
                nrow=2
            )

    # save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "diffusion_half_upscale.pth"))
    print("Model saved.")
    end_time = time.time()
    total_sec = int(end_time - start_time)
    minutes = total_sec // 60
    seconds = total_sec % 60
    print(f"Total training time: {minutes} min {seconds} sec")


# ============================================================
# 6. SAMPLING UTILITIES
# ============================================================

@torch.no_grad()
def p_sample(model, x_t, t):
    """
    One reverse diffusion step: p(x_{t-1} | x_t).
    t: [B] with current timestep values.
    """
    betas_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_ahat_t = extract(sqrt_one_minus_ahat, t, x_t.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)

    # predict noise
    eps_theta = model(x_t, t)

    # compute x0 estimate
    a_hat_t = extract(alpha_hat, t, x_t.shape)
    x0_pred = (x_t - sqrt_one_minus_ahat_t * eps_theta) / (a_hat_t.sqrt() + 1e-8)

    # For simplicity, we can instead use the "simplified" form:
    # mean = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - a_hat_t)) * eps_theta)
    # which is more common and cleaner:
    alpha_t = extract(alphas, t, x_t.shape)
    beta_t = betas_t
    mean = sqrt_recip_alphas_t * (x_t - (beta_t / sqrt_one_minus_ahat_t) * eps_theta)

    # add noise if t > 0
    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
    var = beta_t  # typical DDPM uses beta_t as variance
    sample = mean + nonzero_mask * torch.sqrt(var) * noise
    return sample


@torch.no_grad()
def sample_from_noise(model, n=4):
    """
    Full sampling from pure Gaussian noise.
    """
    img = torch.randn(n, 3, HR_SIZE, HR_SIZE, device=DEVICE)

    for t_step in reversed(range(TIMESTEPS)):
        t = torch.full((n,), t_step, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t)
    return img.clamp(-1, 1)


@torch.no_grad()
def sample_from_partial(model, x_k, k):
    """
    REVERSE DIFFUSION STARTING FROM PARTIALLY NOISED IMAGE x_k.
    x_k is assumed to be an image at timestep k (simulating GAN output).
    We denoise from t = k down to 0.
    """
    if k < 0 or k >= TIMESTEPS:
        raise ValueError(f"k must be in [0, {TIMESTEPS-1}], got {k}")

    img = x_k
    for t_step in reversed(range(k + 1)):  # k, k-1, ..., 0
        t = torch.full((img.shape[0],), t_step, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t)
    return img.clamp(-1, 1)


# ============================================================
# 7. TESTING "K EPOCHS OF NOISE"
# ============================================================

def save_k_noise_examples(k, num=4):
    """
    Use some HR textures, add k steps of noise, and save side-by-side
    to visually inspect what the "GAN-like" noisy input looks like.
    """
    ds = TextureDataset(TEXTURE_DIR, img_size=HR_SIZE, max_images=num)
    dl = DataLoader(ds, batch_size=num, shuffle=False)
    x0 = next(iter(dl)).to(DEVICE)   # [-1,1]
    xk = add_noise_k(x0, k)

    # concat for comparison
    both = torch.cat([
        (x0 + 1) / 2,   # original
        (xk + 1) / 2    # k-step noised
    ], dim=0)

    save_path = os.path.join(NOISE_TESTS_DIR, f"noised_k_{k}.png")
    save_image(both, save_path, nrow=num)
    print(f"Saved k-noise examples at: {save_path}")


# ============================================================
# 8. MAIN EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # --- Quick startup diagnostics ---
    print("========== STARTUP CHECK ==========")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print(f"Texture directory: {TEXTURE_DIR}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Image size: {HR_SIZE}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Channels: {CHANNELS}")
    print("===================================")

    # Count dataset images
    dataset_paths = glob.glob(os.path.join(TEXTURE_DIR, "*"))
    print(f"Found {len(dataset_paths)} texture files for training.\n")


    # 1) Train diffusion model (on HR textures)
    #    Comment this out once you've trained and saved the model
    train()

    # 2) Load model for inference
    model = SimpleUNet(in_ch=3, base_ch=CHANNELS).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "diffusion_half_upscale.pth"), map_location=DEVICE)
    )
    model.eval()

    # 3) Test "k epochs of noise" (simulate GAN output)
    k = TIMESTEPS // 6   # e.g., "halfway" noise, adjust as you like
    test_noise = False
    if test_noise:
        save_k_noise_examples(k, num=4)
        exit()
    else:
        save_k_noise_examples(k, num=4)


    # 4) Take some HR images, add k noise steps, and then denoise from that partial
    ds = TextureDataset(TEXTURE_DIR, img_size=HR_SIZE, max_images=4)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    x0_batch = next(iter(dl)).to(DEVICE)        # ground truth
    xk_batch = add_noise_k(x0_batch, k)        # simulate GAN output

    with torch.no_grad():
        x_denoised = sample_from_partial(model, xk_batch, k)

    # save comparison: original | x_k (GAN-like) | restored
    all_imgs = torch.cat([
        (x0_batch + 1) / 2,
        (xk_batch + 1) / 2,
        (x_denoised + 1) / 2
    ], dim=0)

    save_image(all_imgs, os.path.join(RESULTS_DIR, f"partial_denoise_k_{k}.png"), nrow=4)
    print("Saved partial denoise comparison.")
