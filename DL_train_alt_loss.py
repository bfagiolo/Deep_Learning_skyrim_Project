from tqdm import tqdm
import os
import math
import glob
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, help="the data key of the set. EX: STONE-ARCH for stone and architecture textures.")
args = parser.parse_args()
# ============================================================
# 1. CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXTURE_DIRS = {"LOCALDEBUG": "/Users/slitf/Downloads/stone_masonry/", "NORMALS": "data/png_input/normals/",
                "PLANTS": "data/png_input/diffuse/organized_textures/nature_foliage/",
                "SNOW": "data/png_input/diffuse/organized_textures/snow_ice/",
                "STONE-ARCH": "data/png_input/diffuse/organized_textures/stone_masonry/",
                "TERRAIN": "data/png_input/diffuse/organized_textures/terrain_dirt/",
                "CLOTHING": "data/png_input/diffuse/organized_textures/armors/"}
TEXTURE_DIR = TEXTURE_DIRS[args.d]
SAVE_DIR = "diffusion_run/" + args.d + "/"
MODEL_DIR = os.path.join(SAVE_DIR, "model")
TRAIN_SAMPLES_DIR = os.path.join(SAVE_DIR, "train_samples")
RESULTS_DIR = os.path.join(SAVE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAIN_SAMPLES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEBUG_FAST_RUN = False
DEBUG_NOISE_TEST = False

MAX_TRAINING_NOISE_LEVEL = 0.25
NOISE_SCHEDULE = "cosine"

# ============================================================
# BLUR SCHEDULE CONFIG
# ============================================================
USE_BLUR = True  # Toggle blur on/off
MAX_TRAINING_BLUR_LEVEL = 0.75  # 0.0 = no blur, 1.0 = max blur
BLUR_SCHEDULE = "cosine"  # "linear" or "cosine"
MAX_BLUR_SIGMA = 4.0  # Maximum Gaussian blur sigma at blur_level=1.0
MIN_BLUR_SIGMA = 0.0  # Minimum blur sigma at blur_level=0.0

# Patch training parameters
PATCH_SIZE = 256
PATCHES_PER_IMAGE = 4

if DEBUG_FAST_RUN:
    EPOCHS = 5
    HR_SIZE = 512
    TIMESTEPS = 200
    BATCH_SIZE = 8
    CHANNELS = 64
    MAX_IMAGES = 50
    PATCH_SIZE = 128
else:
    EPOCHS = 100
    HR_SIZE = 2048  # Support 2K textures
    TIMESTEPS = 3000
    BATCH_SIZE = 64  # INCREASED - RTX 6000 has 48GB VRAM
    CHANNELS = 128
    MAX_IMAGES = None
    PATCH_SIZE = 256

LR = 2e-4

# Dataset caching option
CACHE_IMAGES_IN_MEMORY = True  # Set False if you run out of RAM


# ============================================================
# 2. PATCH DATASET
# ============================================================
class PatchTextureDataset(Dataset):
    """
    Dataset that returns random patches from high-res textures.
    Each __getitem__ returns a single patch of fixed size.
    Works with variable-size textures.
    """

    def __init__(self, root_dir, patch_size=512,
                 patches_per_image=4, max_images=None, min_size=None,
                 cache_in_memory=False):
        # Get all image paths
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.paths = [
            p for p in self.paths
            if os.path.isfile(p)
            and p.lower().endswith(('.png', '.jpg', '.jpeg', '.dds'))
        ]

        # Limit number of images if requested
        if max_images is not None:
            self.paths = self.paths[:max_images]

        # Filter out images that are too small
        valid_paths = []
        for p in self.paths:
            try:
                with Image.open(p) as img:
                    valid_paths.append(p)
            except Exception as e:
                print(f"Failed to open image {p}: {e}")
        self.paths = valid_paths

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.min_size = min_size or patch_size

        # Only convert to tensor and normalize - keep native resolution
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])

        # Cache images in memory for faster training
        self.cache_in_memory = cache_in_memory
        self.image_cache = {}

        if cache_in_memory and len(self.paths) > 0:
            print(f"Caching {len(self.paths)} images in memory...")
            for p in tqdm(self.paths, desc="Loading images"):
                try:
                    img = Image.open(p).convert("RGB")
                    self.image_cache[p] = self.transform(img)
                except Exception as e:
                    print(f"Failed to cache {p}: {e}")
            print(f"Cached {len(self.image_cache)} images")

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        # Map idx to image and patch number
        img_idx = idx // self.patches_per_image
        path = self.paths[img_idx]

        # Load from cache or disk
        if self.cache_in_memory and path in self.image_cache:
            img = self.image_cache[path].clone()
        else:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)

        # Extract random patch
        _, h, w = img.shape

        # If image is smaller than min_size, pad it
        if h < self.min_size or w < self.min_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
                _, h, w = img.shape

        # Extract random patch (or full image if still smaller)
        if h >= self.patch_size and w >= self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
        else:
            top, left = 0, 0

        patch = img[:, top:top + self.patch_size, left:left + self.patch_size]

        return patch


# ============================================================
# 3. NOISE SCHEDULE + DIFFUSION UTILITIES
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from:
    Nichol & Dhariwal 2021 (Improved DDPM)

    Produces alphas_cumprod directly via the cosine curve,
    then converts to betas.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=DEVICE)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize

    # Extract betas
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return torch.clip(betas, 0.0001, 0.9999)


betas = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

sqrt_alpha_hat = torch.sqrt(alpha_hat)
sqrt_one_minus_ahat = torch.sqrt(1.0 - alpha_hat)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)


def get_alpha_from_noise_level(noise_level):
    """
    Convert noise level to alpha value directly.

    noise_level: 0.0 to 1.0
    - 0.0 = no noise (alpha = 1.0) = ground truth
    - 1.0 = full noise (alpha = 0.0) = pure Gaussian

    The relationship: alpha = 1 - noise_level
    This gives us: x_noisy = sqrt(alpha)*x0 + sqrt(1-alpha)*noise
    """
    noise_level = max(0.0, min(1.0, noise_level))
    alpha = 1.0 - noise_level
    return alpha


def get_timestep_from_noise_level(noise_level, alpha_hat_tensor, timesteps):
    """
    Find the timestep index that corresponds to the desired noise level.
    This maps our continuous noise level [0,1] to a discrete timestep.

    The schedule (alpha_hat) determines how this mapping works:
    - Linear schedule: roughly linear mapping
    - Cosine schedule: more timesteps spent at low noise levels
    """
    target_alpha = get_alpha_from_noise_level(noise_level)
    diffs = torch.abs(alpha_hat_tensor.cpu() - target_alpha)
    k = torch.argmin(diffs).item()
    return min(k, timesteps - 1)


def extract(a, t, x_shape):
    b = a.gather(-1, t)
    return b.reshape(-1, 1, 1, 1)


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a_hat = extract(sqrt_alpha_hat, t, x0.shape)
    om_a = extract(sqrt_one_minus_ahat, t, x0.shape)
    return a_hat * x0 + om_a * noise


def add_noise_k(x0, k):
    """
    ADD K STEPS OF NOISE using the schedule.
    k is an integer 0 <= k < TIMESTEPS that indexes into our schedule.
    """
    k = min(k, TIMESTEPS - 1)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    bsz = x0.shape[0]
    t = torch.full((bsz,), k, device=x0.device, dtype=torch.long)
    x_k = q_sample(x0, t)
    return x_k


def add_noise_level(x0, noise_level):
    """
    Add noise based on noise level directly, independent of timesteps.

    noise_level: 0.0 to 1.0
    - 0.0 = no noise (returns x0) = ground truth
    - 1.0 = full noise (returns pure Gaussian)

    Uses: x_noisy = sqrt(alpha) * x0 + sqrt(1-alpha) * noise
    where alpha = 1 - noise_level

    This is independent of the schedule - it's a direct noise injection.
    """
    alpha = get_alpha_from_noise_level(noise_level)

    if alpha >= 1.0:
        return x0
    if alpha <= 0.0:
        return torch.randn_like(x0)

    noise = torch.randn_like(x0)
    sqrt_alpha = math.sqrt(alpha)
    sqrt_one_minus_alpha = math.sqrt(1.0 - alpha)

    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


# ============================================================
# 3b. BLUR SCHEDULE + BLUR UTILITIES
# ============================================================

def get_blur_sigma_from_level(blur_level, schedule="linear"):
    """
    Convert blur level to Gaussian sigma.

    blur_level: 0.0 to 1.0
    - 0.0 = no blur (sigma = MIN_BLUR_SIGMA)
    - 1.0 = max blur (sigma = MAX_BLUR_SIGMA)

    schedule: "linear" or "cosine"
    - linear: sigma scales linearly with blur_level
    - cosine: more gradual blur at low levels (like noise cosine schedule)
    """
    blur_level = max(0.0, min(1.0, blur_level))

    if schedule == "cosine":
        # Cosine interpolation - slower start, faster end
        # Maps [0,1] -> [0,1] with cosine curve
        t = (1 - math.cos(blur_level * math.pi)) / 2
    else:
        # Linear interpolation
        t = blur_level

    sigma = MIN_BLUR_SIGMA + t * (MAX_BLUR_SIGMA - MIN_BLUR_SIGMA)
    return sigma


def get_blur_kernel_size(sigma):
    """
    Calculate appropriate kernel size for given sigma.
    Kernel size should be odd and large enough to capture the Gaussian.
    Rule of thumb: kernel_size = 6*sigma + 1 (rounded to odd)
    """
    if sigma <= 0:
        return 1
    kernel_size = int(6 * sigma + 1)
    # Ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Minimum size of 3
    kernel_size = max(3, kernel_size)
    return kernel_size


def create_gaussian_kernel(kernel_size, sigma, device):
    """
    Create a 2D Gaussian kernel for convolution.
    """
    if sigma <= 0:
        # Return identity kernel (no blur)
        kernel = torch.zeros(kernel_size, kernel_size, device=device)
        center = kernel_size // 2
        kernel[center, center] = 1.0
        return kernel

    # Create 1D Gaussian
    x = torch.arange(kernel_size, device=device, dtype=torch.float32)
    x = x - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))

    # Create 2D kernel via outer product
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]

    # Normalize
    kernel_2d = kernel_2d / kernel_2d.sum()

    return kernel_2d


def apply_gaussian_blur(x, sigma):
    """
    Apply Gaussian blur to tensor x with given sigma.

    x: [B, C, H, W] tensor in range [-1, 1]
    sigma: blur strength (0 = no blur)

    Returns blurred tensor in same range.
    """
    if sigma <= 0:
        return x

    kernel_size = get_blur_kernel_size(sigma)
    kernel = create_gaussian_kernel(kernel_size, sigma, x.device)

    # Expand kernel for depthwise convolution [out_ch, in_ch/groups, kH, kW]
    # For depthwise: out_ch = in_ch, groups = in_ch
    channels = x.shape[1]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    # Padding to maintain spatial size
    padding = kernel_size // 2

    # Apply depthwise convolution
    blurred = F.conv2d(x, kernel, padding=padding, groups=channels)

    return blurred


def add_blur_level(x0, blur_level, schedule="linear"):
    """
    Add blur based on blur level.

    blur_level: 0.0 to 1.0
    - 0.0 = no blur (returns x0)
    - 1.0 = maximum blur

    schedule: "linear" or "cosine" for sigma interpolation
    """
    if blur_level <= 0:
        return x0

    sigma = get_blur_sigma_from_level(blur_level, schedule)
    return apply_gaussian_blur(x0, sigma)


def add_noise_and_blur(x0, noise_level, blur_level, blur_schedule="linear"):
    """
    Add both noise and blur to image.

    Order: Blur first, then add noise.
    This simulates degraded input that needs both deblurring and denoising.

    x0: [B, C, H, W] clean image
    noise_level: 0.0 to 1.0
    blur_level: 0.0 to 1.0

    Returns degraded image.
    """
    # Apply blur first
    x_blurred = add_blur_level(x0, blur_level, blur_schedule)

    # Then add noise
    x_degraded = add_noise_level(x_blurred, noise_level)

    return x_degraded


# ============================================================
# 4. ENHANCED UNET WITH ATTENTION
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.linspace(0, 1, half, device=t.device)
        )
        t = t.float().unsqueeze(1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

        # Adaptive group count - find largest divisor <= 8
        def get_num_groups(channels):
            for g in [8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1

        self.norm1 = nn.GroupNorm(get_num_groups(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(get_num_groups(out_ch), out_ch)

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding
        temb = self.time_mlp(t_emb)
        h = h + temb[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention for texture patterns"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.reshape(b, c, h * w).transpose(1, 2)  # [B, HW, C]
        k = k.reshape(b, c, h * w).transpose(1, 2)
        v = v.reshape(b, c, h * w).transpose(1, 2)

        # Scaled dot-product attention
        scale = c ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(1, 2).reshape(b, c, h, w)
        out = self.proj(out)

        return out + residual


class TextureUNet(nn.Module):
    """Enhanced U-Net optimized for texture detail"""

    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # Encoder
        self.down1 = nn.ModuleList([
            ResidualBlock(in_ch, base_ch, time_emb_dim),
            ResidualBlock(base_ch, base_ch, time_emb_dim)
        ])

        self.down2 = nn.ModuleList([
            ResidualBlock(base_ch, base_ch * 2, time_emb_dim),
            ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        ])

        self.down3 = nn.ModuleList([
            ResidualBlock(base_ch * 2, base_ch * 4, time_emb_dim),
            ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        ])

        # Bottleneck with attention
        self.mid = nn.ModuleList([
            ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim),
            AttentionBlock(base_ch * 4),
            ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        ])

        # Decoder
        self.up3 = nn.ModuleList([
            ResidualBlock(base_ch * 4 + base_ch * 4, base_ch * 4, time_emb_dim),
            ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim),
            AttentionBlock(base_ch * 2)
        ])

        self.up2 = nn.ModuleList([
            ResidualBlock(base_ch * 2 + base_ch * 2, base_ch * 2, time_emb_dim),
            ResidualBlock(base_ch * 2, base_ch, time_emb_dim)
        ])

        self.up1 = nn.ModuleList([
            ResidualBlock(base_ch + base_ch, base_ch, time_emb_dim),
            ResidualBlock(base_ch, base_ch, time_emb_dim)
        ])

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # Encoder
        h1 = x
        for block in self.down1:
            h1 = block(h1, t_emb)

        h2 = self.pool(h1)
        for block in self.down2:
            h2 = block(h2, t_emb)

        h3 = self.pool(h2)
        for block in self.down3:
            h3 = block(h3, t_emb)

        # Bottleneck
        h = self.pool(h3)
        for block in self.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)

        # Decoder
        h = self.upsample(h)
        h = torch.cat([h, h3], dim=1)
        for block in self.up3:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)

        h = self.upsample(h)
        h = torch.cat([h, h2], dim=1)
        for block in self.up2:
            h = block(h, t_emb)

        h = self.upsample(h)
        h = torch.cat([h, h1], dim=1)
        for block in self.up1:
            h = block(h, t_emb)

        return self.final(h)


# ============================================================
# 5. TRAINING WITH HIGH-FREQUENCY LOSS
# ============================================================

def high_frequency_loss(pred, target):
    """Laplacian-based high-frequency loss"""
    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 4.0

    kernel = kernel.repeat(pred.shape[1], 1, 1, 1)

    pred_hf = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    target_hf = F.conv2d(target, kernel, padding=1, groups=target.shape[1])

    return F.l1_loss(pred_hf, target_hf)


def p_losses(
    model,
    x0,
    t,
    mse_weight=1.0,
    recon_weight=0.5,
    hf_weight=0.5,
    use_blur=False,
    blur_level=0.0,
):
    """
    Combined loss:
    - noise MSE (standard diffusion training)
    - image L1 between reconstructed x0_pred and clean x0
    - high-frequency loss between x0_pred and clean x0
    """
    noise = torch.randn_like(x0)

    # Apply blur to x0 before adding noise if enabled
    if use_blur and blur_level > 0:
        x0_processed = add_blur_level(x0, blur_level, BLUR_SCHEDULE)
    else:
        x0_processed = x0

    # Forward diffusion: q(x_t | x0_processed, t)
    x_t = q_sample(x0_processed, t, noise)

    # Predict noise
    noise_pred = model(x_t, t)

    # 1) Standard MSE on noise
    mse_loss = F.mse_loss(noise_pred, noise)

    # 2) Reconstruct x0 from predicted noise
    a_hat_t = extract(sqrt_alpha_hat, t, x_t.shape)
    sqrt_one_minus_ahat_t = extract(sqrt_one_minus_ahat, t, x_t.shape)

    # x0_pred = (x_t - sqrt(1 - alpha_hat_t) * eps_theta) / sqrt(alpha_hat_t)
    x0_pred = (x_t - sqrt_one_minus_ahat_t * noise_pred) / a_hat_t
    x0_pred = x0_pred.clamp(-1.0, 1.0)

    # 3) Image-space reconstruction (to keep colors/structure)
    recon_loss = F.l1_loss(x0_pred, x0)

    # 4) High-frequency loss directly on reconstructed images
    hf_loss = high_frequency_loss(x0_pred, x0)

    total_loss = (
        mse_weight * mse_loss
        + recon_weight * recon_loss
        + hf_weight * hf_loss
    )

    return (
        total_loss,
        mse_loss.item(),
        recon_loss.item(),
        hf_loss.item(),
    )


def train():
    start_time = time.time()

    # Use patch dataset with caching
    dataset = PatchTextureDataset(
        TEXTURE_DIR,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        max_images=MAX_IMAGES,
        cache_in_memory=CACHE_IMAGES_IN_MEMORY
    )
    print(f"Using PATCH training: {PATCH_SIZE}x{PATCH_SIZE} patches")
    print(f"Native resolution textures - no downsampling")

    # Optimized DataLoader
    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    model = TextureUNet(in_ch=3, base_ch=CHANNELS).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    k_max = get_timestep_from_noise_level(
        MAX_TRAINING_NOISE_LEVEL, alpha_hat, TIMESTEPS
    )

    print(f"Training with patch-based approach")
    print(f"Patch sizes: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Noise schedule: {NOISE_SCHEDULE}")
    print(f"Max training noise level: {MAX_TRAINING_NOISE_LEVEL:.2f} (timestep k={k_max})")
    if USE_BLUR:
        blur_sigma = get_blur_sigma_from_level(MAX_TRAINING_BLUR_LEVEL, BLUR_SCHEDULE)
        print(f"Blur ENABLED: max level={MAX_TRAINING_BLUR_LEVEL:.2f}, sigma={blur_sigma:.2f}")
        print(f"Blur schedule: {BLUR_SCHEDULE}")
    else:
        print(f"Blur DISABLED")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dl)}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_recon = 0.0
        epoch_hf = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for x0 in pbar:
            x0 = x0.to(DEVICE)

            t = torch.randint(0, k_max + 1, (x0.shape[0],), device=DEVICE).long()

            # Sample random blur level for this batch (if enabled)
            if USE_BLUR:
                batch_blur_level = random.uniform(0, MAX_TRAINING_BLUR_LEVEL)
            else:
                batch_blur_level = 0.0

            loss, mse, recon, hf = p_losses(
                model,
                x0,
                t,
                mse_weight=1.0,
                recon_weight=0.5,
                hf_weight=0.5,
                use_blur=USE_BLUR,
                blur_level=batch_blur_level,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_mse += mse
            epoch_recon += recon
            epoch_hf += hf

            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(dl)
        avg_mse = epoch_mse / len(dl)
        avg_recon = epoch_recon / len(dl)
        avg_hf = epoch_hf / len(dl)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} "
            f"(MSE: {avg_mse:.4f}, Recon: {avg_recon:.4f}, HF: {avg_hf:.4f})"
        )

        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x_test = next(iter(dl))[:4].to(DEVICE)
                k_test = k_max // 2

                # Apply blur if enabled
                if USE_BLUR:
                    x_test_processed = add_blur_level(x_test, MAX_TRAINING_BLUR_LEVEL / 2, BLUR_SCHEDULE)
                else:
                    x_test_processed = x_test

                x_noisy = add_noise_k(x_test_processed, k_test)

                # Quick denoising visualization
                t_test = torch.full((4,), k_test, device=DEVICE, dtype=torch.long)
                x_pred = model(x_noisy, t_test)

                comparison = torch.cat([
                    (x_noisy + 1) / 2,
                    (x_test + 1) / 2,
                    (x_pred + 1) / 2
                ], dim=0)

                save_image(
                    comparison,
                    os.path.join(TRAIN_SAMPLES_DIR, f"epoch_{epoch + 1}.png"),
                    nrow=4
                )

    # Save model
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, "texture_diffusion.pth"))

    end_time = time.time()
    total_sec = int(end_time - start_time)
    print(f"\nTraining completed in {total_sec // 60}m {total_sec % 60}s")


# ============================================================
# 6. SAMPLING
# ============================================================

@torch.no_grad()
def p_sample(model, x_t, t):
    betas_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_ahat_t = extract(sqrt_one_minus_ahat, t, x_t.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)

    eps_theta = model(x_t, t)

    mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_ahat_t) * eps_theta)

    if t[0] > 0:
        noise = torch.randn_like(x_t)
        var = betas_t
        sample = mean + torch.sqrt(var) * noise
    else:
        sample = mean

    return sample


@torch.no_grad()
def sample_from_partial(model, x_k, k):
    k = min(k, TIMESTEPS - 1)
    img = x_k

    for t_step in reversed(range(k + 1)):
        t = torch.full((img.shape[0],), t_step, device=DEVICE, dtype=torch.long)
        img = p_sample(model, img, t)

    return img.clamp(-1, 1)


# ============================================================
# 7. VISUALIZATION WITH NOISE AND BLUR
# ============================================================

def visualize_starting_noise(noise_level, num=4, tag="check",
                             blur_level=None, use_blur=None):
    """
    Saves a comparison of Original vs Degraded images at the given noise/blur levels.

    noise_level: 0.0 to 1.0
    - 0.0 = clean (ground truth)
    - 1.0 = pure noise

    blur_level: 0.0 to 1.0 (optional, defaults to MAX_TRAINING_BLUR_LEVEL if use_blur=True)
    - 0.0 = no blur
    - 1.0 = max blur

    use_blur: bool (optional, defaults to global USE_BLUR)

    Output image layout (when blur enabled):
    - Row 1: Original
    - Row 2: Blurred only
    - Row 3: Noised only
    - Row 4: Blurred + Noised

    Output image layout (when blur disabled):
    - Row 1: Original
    - Row 2: Noised
    """
    # Use defaults from config if not specified
    if use_blur is None:
        use_blur = USE_BLUR
    if blur_level is None:
        blur_level = MAX_TRAINING_BLUR_LEVEL if use_blur else 0.0

    # Use patch dataset for visualization
    ds = PatchTextureDataset(TEXTURE_DIR, patch_size=PATCH_SIZE,
                             patches_per_image=1, max_images=num)
    if len(ds) == 0:
        print("No images found for visualization.")
        return

    dl = DataLoader(ds, batch_size=num, shuffle=False)
    x0 = next(iter(dl)).to(DEVICE)  # [-1,1]

    # Get timestep info for noise
    k = get_timestep_from_noise_level(noise_level, alpha_hat, TIMESTEPS)

    if use_blur and blur_level > 0:
        # Generate all four versions
        x_original = x0
        x_blurred = add_blur_level(x0, blur_level, BLUR_SCHEDULE)
        x_noised = add_noise_level(x0, noise_level)
        x_blur_noise = add_noise_and_blur(x0, noise_level, blur_level, BLUR_SCHEDULE)

        # Get blur sigma for logging
        blur_sigma = get_blur_sigma_from_level(blur_level, BLUR_SCHEDULE)

        # Stack all rows
        all_imgs = torch.cat([
            (x_original + 1) / 2,  # Row 1: Original
            (x_blurred + 1) / 2,  # Row 2: Blur only
            (x_noised + 1) / 2,  # Row 3: Noise only
            (x_blur_noise + 1) / 2,  # Row 4: Blur + Noise
        ], dim=0)

        save_path = os.path.join(
            RESULTS_DIR,
            f"degradation_vis_noise{noise_level:.2f}_blur{blur_level:.2f}_{tag}.png"
        )
        save_image(all_imgs, save_path, nrow=num)

        print(f"Saved visualization: {save_path}")
        print(f"  Noise level: {noise_level:.2f} (timestep k={k})")
        print(f"  Blur level: {blur_level:.2f} (sigma={blur_sigma:.2f})")
        print(f"  Rows: Original | Blur only | Noise only | Blur+Noise")

    else:
        # Original behavior - just noise
        xk = add_noise_level(x0, noise_level)

        # Stack: Top row = Original, Bottom row = Noised
        both = torch.cat([
            (x0 + 1) / 2,  # original
            (xk + 1) / 2  # noised
        ], dim=0)

        save_path = os.path.join(RESULTS_DIR, f"noise_vis_level{noise_level:.2f}_{tag}.png")
        save_image(both, save_path, nrow=num)

        print(f"Saved visualization: {save_path}")
        print(f"  Noise level: {noise_level:.2f} (timestep k={k})")
        print(f"  Rows: Original | Noised")


def visualize_blur_schedule(num_levels=5, num_images=4, tag="blur_schedule"):
    """
    Visualize the blur schedule at different levels.
    Creates a grid showing how blur progresses from 0 to MAX_TRAINING_BLUR_LEVEL.
    """
    ds = PatchTextureDataset(TEXTURE_DIR, patch_size=PATCH_SIZE,
                             patches_per_image=1, max_images=num_images)
    if len(ds) == 0:
        print("No images found for visualization.")
        return

    dl = DataLoader(ds, batch_size=num_images, shuffle=False)
    x0 = next(iter(dl)).to(DEVICE)

    blur_levels = [i * MAX_TRAINING_BLUR_LEVEL / (num_levels - 1) for i in range(num_levels)]

    rows = [(x0 + 1) / 2]  # Start with original

    print(f"Blur schedule visualization ({BLUR_SCHEDULE}):")
    for bl in blur_levels[1:]:  # Skip 0.0 since we already have original
        sigma = get_blur_sigma_from_level(bl, BLUR_SCHEDULE)
        x_blurred = add_blur_level(x0, bl, BLUR_SCHEDULE)
        rows.append((x_blurred + 1) / 2)
        print(f"  Level {bl:.2f}: sigma={sigma:.2f}")

    all_imgs = torch.cat(rows, dim=0)
    save_path = os.path.join(RESULTS_DIR, f"{tag}.png")
    save_image(all_imgs, save_path, nrow=num_images)
    print(f"Saved: {save_path}")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PATCH-BASED TEXTURE DIFFUSION UPSCALER")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Timesteps: {TIMESTEPS} (sampling granularity)")
    print(f"Noise Schedule: {NOISE_SCHEDULE}")
    print(f"Base Channels: {CHANNELS}")
    print(f"Max Training Noise Level: {MAX_TRAINING_NOISE_LEVEL}")

    # Print blur config
    print("-" * 40)
    print(f"Blur Enabled: {USE_BLUR}")
    if USE_BLUR:
        print(f"Max Training Blur Level: {MAX_TRAINING_BLUR_LEVEL}")
        print(f"Blur Schedule: {BLUR_SCHEDULE}")
        print(f"Blur Sigma Range: {MIN_BLUR_SIGMA} to {MAX_BLUR_SIGMA}")
        max_sigma = get_blur_sigma_from_level(MAX_TRAINING_BLUR_LEVEL, BLUR_SCHEDULE)
        print(f"Max Training Blur Sigma: {max_sigma:.2f}")
    print("-" * 40)

    dataset_paths = glob.glob(os.path.join(TEXTURE_DIR, "*"))
    dataset_paths = [p for p in dataset_paths if os.path.isfile(p)
                     and p.lower().endswith(('.png', '.jpg', '.jpeg', '.dds'))]
    print(f"\nFound {len(dataset_paths)} texture files")

    # Calculate which timestep corresponds to our noise level
    k_start = get_timestep_from_noise_level(
        MAX_TRAINING_NOISE_LEVEL, alpha_hat, TIMESTEPS
    )
    print(f"Noise level {MAX_TRAINING_NOISE_LEVEL:.2f} maps to timestep k={k_start}/{TIMESTEPS}")
    print("=" * 60)

    # TEST STARTING NOISE/BLUR LEVEL FIRST
    visualize_starting_noise(
        MAX_TRAINING_NOISE_LEVEL,
        num=4,
        tag="pre_train",
        blur_level=MAX_TRAINING_BLUR_LEVEL if USE_BLUR else 0.0,
        use_blur=USE_BLUR
    )

    # Also visualize the blur schedule if enabled
    if USE_BLUR:
        visualize_blur_schedule(num_levels=5, num_images=4, tag="blur_schedule_test")

    if DEBUG_NOISE_TEST:
        print(f"\n[DEBUG] Visualizing noise level {MAX_TRAINING_NOISE_LEVEL}")
        if USE_BLUR:
            print(f"[DEBUG] Visualizing blur level {MAX_TRAINING_BLUR_LEVEL}")
        exit()

    # Train
    train()

    # Load and test
    print("\nLoading model for testing...")
    model = TextureUNet(in_ch=3, base_ch=CHANNELS).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "texture_diffusion.pth"),
                   map_location=DEVICE)
    )
    model.eval()

    # Test denoising on patches
    test_dataset = PatchTextureDataset(TEXTURE_DIR, patch_size=PATCH_SIZE,
                                       patches_per_image=1, max_images=4)
    test_dl = DataLoader(test_dataset, batch_size=4, shuffle=False)
    x0_batch = next(iter(test_dl)).to(DEVICE)

    # Get timestep for our noise level
    k = get_timestep_from_noise_level(
        MAX_TRAINING_NOISE_LEVEL, alpha_hat, TIMESTEPS
    )

    print(f"Denoising from noise level {MAX_TRAINING_NOISE_LEVEL:.2f} (k={k}) to 0.0...")

    # Apply blur + noise if enabled
    if USE_BLUR:
        xk_batch = add_noise_and_blur(x0_batch, MAX_TRAINING_NOISE_LEVEL,
                                      MAX_TRAINING_BLUR_LEVEL, BLUR_SCHEDULE)
        print(f"Also deblurring from blur level {MAX_TRAINING_BLUR_LEVEL:.2f}")
    else:
        xk_batch = add_noise_k(x0_batch, k)

    x_denoised = sample_from_partial(model, xk_batch, k)

    all_imgs = torch.cat([
        (x0_batch + 1) / 2,  # Original patches (clean)
        (xk_batch + 1) / 2,  # Degraded (noise + optional blur)
        (x_denoised + 1) / 2  # Denoised/deblurred result
    ], dim=0)

    save_image(all_imgs, os.path.join(RESULTS_DIR, "final_comparison.png"), nrow=4)
    print(f"\nResults saved to {RESULTS_DIR}")
    if USE_BLUR:
        print(f"Rows: Original | Degraded (noise+blur) | Restored")
    else:
        print(f"Rows: Original (0.0) | Noised ({MAX_TRAINING_NOISE_LEVEL:.2f}) | Denoised")
