import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
import os
import glob
import sys
from tqdm import tqdm

from gan import GAN
from DL_train import add_noise_level

DEBUG = False
CONFIG = {
    'resolution': 256,
    'scale': 4,
    'batch_size': 32,
    'lr': 0.002,
    'epochs': 100,
    'sample_interval': 5,
    'sample_path': './gan_run/samples',
    'checkpoint_path': './gan_run',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_path': './png_input/diffuse/organized_textures/terrain_dirt'
}


class CachedTextureDataset(Dataset):
    def __init__(self, root_dir, scale, resolution):
        self.files = glob.glob(os.path.join(root_dir, '*.png')) + \
                     glob.glob(os.path.join(root_dir, '*.jpg'))
        self.scale = scale
        self.resolution = resolution
        self.data_cache = []
        self.patches_per_img = 16
        # TRANSFORMS (Random Crop 256x256)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        print(f"Found {len(self.files)} images. Preloading to RAM...")
        if len(self.files) == 0:
            print("WARNING: No images found.")

        for file_path in tqdm(self.files, desc="Caching Dataset"):
            try:
                img = Image.open(file_path).convert('RGB')
                for i in range(self.patches_per_img):
                    crop = transforms.RandomCrop((resolution, resolution), pad_if_needed=True, padding_mode='replicate')
                    patch = self.transform(crop(img))
                    self.data_cache.append(patch)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        high_res = self.data_cache[idx]

        # downscale input and then resize og res to mimic outdated texture
        lr_size = self.resolution // self.scale
        low_res = F.interpolate(high_res.unsqueeze(0), size=(lr_size, lr_size), mode='bilinear',
                                align_corners=False).squeeze(0)

        # create target: high Res Noisy (256x256)
        target_noisy = add_noise_level(model(high_res))

        return low_res, target_noisy


def train():
    device = CONFIG['device']
    os.makedirs(CONFIG['sample_path'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_path'], exist_ok=True)

    print(f"Initializing Stage 1 GAN on {device}...")
    model = OMRAT_GAN(resolution=CONFIG['resolution']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.99))
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    if not os.path.exists(CONFIG['image_path']):
        print(f"Error: Path '{CONFIG['image_path']}' does not exist.")
        return

    dataset = CachedTextureDataset(CONFIG['image_path'], CONFIG['scale'], CONFIG['resolution'])
    if len(dataset) == 0: return

    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    # validation batch
    try:
        fixed_low_res, fixed_target = next(iter(dataloader))
        fixed_low_res = fixed_low_res.to(device)
        fixed_target = fixed_target.to(device)
    except StopIteration:
        return

    print(f"Starting training loop...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

        for i, (low_res, target_noisy) in pbar:
            low_res = low_res.to(device)
            target_noisy = target_noisy.to(device)

            optimizer.zero_grad()
            prediction = model(low_res)

            loss = criterion_mse(prediction, target_noisy) + criterion_l1(prediction, target_noisy)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 5 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        if epoch % CONFIG['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                fixed_recon = add_noise_level(model(fixed_low_res))
                fixed_low_res_vis = F.interpolate(fixed_low_res, size=(CONFIG['resolution'], CONFIG['resolution']),
                                                  mode='nearest')

                n_show = min(4, fixed_low_res.size(0))
                grid_img = torch.cat([fixed_low_res_vis[:n_show], fixed_target[:n_show], fixed_recon[:n_show]], dim=0)
                vutils.save_image(grid_img, f"{CONFIG['sample_path']}/sample_epoch_{epoch}.png", nrow=n_show,
                                  normalize=True)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{CONFIG['checkpoint_path']}/gan_epoch_{epoch}.pth")
            os.remove(f"{CONFIG['checkpoint_path']}/gan_epoch_{epoch}.pth")

    torch.save(model.state_dict(), f"{CONFIG['checkpoint_path']}/gan_final_terrain.pth")
    print("Training Complete.")


def gan_inference_patch_based(model, img_path, patch_size, scale, device, results_dir):
    """
    Performs patch-based inference:
    1. Loads HR image.
    2. Downscales it to create a synthetic LR input.
    3. Splits LR input into patches.
    4. Upscales patches using the GAN.
    5. Stitches patches back together.
    """
    import math

    # 1. Load Image & Prepare
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Calculate LR dimensions
    # We round to nearest divisible by scale to ensure clean math
    lr_w = w // scale
    lr_h = h // scale

    # Resize to create the Low-Res Input (This is what the GAN actually sees)
    # We use Bicubic for downscaling as a standard degradation
    img_lr = img.resize((lr_w, lr_h), Image.BICUBIC)

    # Convert to Tensor
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(img_lr).unsqueeze(0).to(device)  # (1, 3, lr_h, lr_w)

    # 2. Pad LR image to match patch stride
    # The model expects inputs of ~64x64 (patch_size // scale)
    lr_patch_size = patch_size // scale

    pad_w = (lr_patch_size - (lr_w % lr_patch_size)) % lr_patch_size
    pad_h = (lr_patch_size - (lr_h % lr_patch_size)) % lr_patch_size

    # Pad (Left, Right, Top, Bottom)
    lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    # 3. Unfold into patches
    # Shape: (1, C, H, W) -> Unfold -> Extract patches
    # This creates a sliding window view. Step equals size for non-overlapping.
    kernel_size = lr_patch_size
    stride = lr_patch_size

    patches = lr_padded.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # patches shape: (1, C, n_rows, n_cols, k_h, k_w)

    n_rows = patches.shape[2]
    n_cols = patches.shape[3]

    # Reshape for batch processing: (n_rows * n_cols, C, k_h, k_w)
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_size, kernel_size)

    # 4. Batched Inference
    model.eval()
    output_patches = []
    batch_size = 16  # Adjust based on VRAM

    print(f"Processing {len(patches)} patches for {filename}...")

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]

            # GAN Forward Pass
            # The model internally upscales 64->256
            # We use inference() if available or forward()
            # Based on your gan.py, model(x) handles the upscale
            generated = model(batch)
            output_patches.append(generated.cpu())

    # Concatenate all batches
    output_patches = torch.cat(output_patches, dim=0)  # (Total_Patches, 3, 256, 256)

    # 5. Stitching (Fold logic manually)
    # We need to reshape back to (n_rows, n_cols, C, H_out, W_out)
    out_h, out_w = patch_size, patch_size  # 256, 256

    output_patches = output_patches.view(n_rows, n_cols, 3, out_h, out_w)

    # Permute to (3, n_rows, out_h, n_cols, out_w)
    # Then flatten the spatial dimensions
    recon_img = output_patches.permute(2, 0, 3, 1, 4).reshape(3, n_rows * out_h, n_cols * out_w)

    # 6. Crop padding
    # The output is scaled by `scale` (4x). We need to remove the padded area.
    final_h = lr_h * scale
    final_w = lr_w * scale
    recon_img = recon_img[:, :final_h, :final_w]

    # 7. Save Visualization
    # Save a grid: [LR (Nearest Neighbor upscaled), Original HR, Reconstructed HR]
    save_path = os.path.join(results_dir, f"test_result_{filename}")

    # Resize LR to HR size for visualization
    lr_vis = F.interpolate(lr_tensor, size=(final_h, final_w), mode='nearest').cpu().squeeze(0)
    hr_vis = to_tensor(img).to(device)

    # Ensure HR matches dimensions (in case of slight rounding errors in loading)
    if hr_vis.shape[1:] != recon_img.shape[1:]:
        hr_vis = transforms.CenterCrop((final_h, final_w))(hr_vis)

    # Clamp results to valid image range
    recon_img = torch.clamp(recon_img, 0, 1)
    noise_recon_img = add_noise_level(recon_img, .1)

    if DEBUG:
        # Create comparison grid
        comparison = torch.cat([lr_vis, hr_vis.cpu(), recon_img, noise_recon_img], dim=2)  # Concatenate horizontally
        vutils.save_image(comparison, save_path)

    vutils.save_image(noise_recon_img, os.path.join(results_dir, f"{filename}"))

    print(f"Saved inference result to {save_path}")

    return recon_img


def gan_test():
    device = CONFIG['device']
    results_dir = CONFIG['sample_path']
    os.makedirs(results_dir, exist_ok=True)

    model = OMRAT_GAN(resolution=CONFIG['resolution']).to(device)
    checkpoint_path = os.path.join(CONFIG['checkpoint_path'], "gan_final.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        return

    model.eval()

    image_paths = glob.glob(os.path.join(CONFIG['image_path'], "*"))
    image_paths = [
        p for p in image_paths
        if os.path.isfile(p)
           and p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        print(f"No images found in {CONFIG['image_path']}")
        return

    # use the first image found
    hr_img_path = image_paths[1]
    x_gan_hr = gan_inference_patch_based(
        model,
        hr_img_path,
        patch_size=CONFIG['resolution'],  # 256
        scale=CONFIG['scale'],  # 4
        device=device,
        results_dir=results_dir
    )


def gan_inference_patch_based_rec(model, img_path, patch_size, scale, device, results_dir):
    """
    Performs patch-based inference:
    1. Loads HR image.
    2. Downscales it to create a synthetic LR input.
    3. Splits LR input into patches.
    4. Upscales patches using the GAN.
    5. Stitches patches back together.

    The 'results_dir' here is the FULL path where the file should be saved,
    including any necessary subdirectories.
    """
    # NOTE: The original logic for saving needs a slight modification.
    # 'results_dir' is now the full save path, not just a base directory.
    # 1. Load Image & Prepare
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Calculate LR dimensions
    lr_w = w // scale
    lr_h = h // scale

    # Resize to create the Low-Res Input
    img_lr = img.resize((lr_w, lr_h), Image.BICUBIC)

    # Convert to Tensor
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(img_lr).unsqueeze(0).to(device)  # (1, 3, lr_h, lr_w)

    # 2. Pad LR image to match patch stride
    lr_patch_size = patch_size // scale
    pad_w = (lr_patch_size - (lr_w % lr_patch_size)) % lr_patch_size
    pad_h = (lr_patch_size - (lr_h % lr_patch_size)) % lr_patch_size

    # Pad (Left, Right, Top, Bottom)
    lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='replicate')

    # 3. Unfold into patches
    kernel_size = lr_patch_size
    stride = lr_patch_size

    patches = lr_padded.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    n_rows = patches.shape[2]
    n_cols = patches.shape[3]

    # Reshape for batch processing
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, 3, kernel_size, kernel_size)

    # 4. Batched Inference
    model.eval()
    output_patches = []
    batch_size = 16

    print(f"Processing {len(patches)} patches for {filename}...")

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            generated = model(batch)
            output_patches.append(generated.cpu())

    # Concatenate all batches
    output_patches = torch.cat(output_patches, dim=0)

    # 5. Stitching
    out_h, out_w = patch_size, patch_size
    output_patches = output_patches.view(n_rows, n_cols, 3, out_h, out_w)
    recon_img = output_patches.permute(2, 0, 3, 1, 4).reshape(3, n_rows * out_h, n_cols * out_w)

    # 6. Crop padding
    final_h = lr_h * scale
    final_w = lr_w * scale
    recon_img = recon_img[:, :final_h, :final_w]

    # 7. Save Visualization
    # The original results_dir is now the FULL save path (e.g., /output/A/a.png)
    save_base, save_filename = os.path.split(results_dir)
    os.makedirs(save_base, exist_ok=True)  # Ensure the subfolder structure exists

    # Resize LR to HR size for visualization
    lr_vis = F.interpolate(lr_tensor, size=(final_h, final_w), mode='nearest').cpu().squeeze(0)
    hr_vis = to_tensor(img).to(device)

    # Ensure HR matches dimensions
    if hr_vis.shape[1:] != recon_img.shape[1:]:
        hr_vis = transforms.CenterCrop((final_h, final_w))(hr_vis)

    # Clamp results to valid image range
    recon_img = torch.clamp(recon_img, 0, 1)
    # The original code had a second save path logic, which we'll adapt:
    # 1. Save the version with noise applied (this seems to be the main output)
    noise_recon_img = add_noise_level(recon_img, .1)
    vutils.save_image(noise_recon_img, results_dir)

    # 2. Save the comparison grid (if DEBUG is True)
    if DEBUG:
        comparison_save_path = os.path.join(save_base, f"test_result_{save_filename}")
        comparison = torch.cat([lr_vis, hr_vis.cpu(), recon_img, noise_recon_img], dim=2)
        vutils.save_image(comparison, comparison_save_path)
        print(f"Saved comparison result to {comparison_save_path}")

    print(f"Saved inference result to {results_dir}")

    return recon_img


# ---
# REVISED gan_test function
# ---

def gan_test_rec():
    """
    Updated to recursively find all image files and save the output
    to a corresponding path that preserves the original subfolder structure.
    """
    device = CONFIG['device']
    input_base_dir = "./textures/skyrim-inference-data/architecture"
    output_base_dir = "./texture_out/textures/architecture"
    os.makedirs(output_base_dir, exist_ok=True)  # Ensure the base output folder exists

    # 1. Load model and checkpoint (UNMODIFIED)
    model = OMRAT_GAN(resolution=CONFIG['resolution']).to(device)
    checkpoint_path = os.path.join("./gan_run/", "gan_final_archit.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        return

    model.eval()

    # 2. Recursively find all image paths
    # The ** is for recursive globbing in Python 3.5+
    search_pattern = os.path.join(input_base_dir, "**", "*")

    # Filter files recursively based on supported extensions
    image_paths = [
        p for p in glob.glob(search_pattern, recursive=True)
        if os.path.isfile(p)
           and p.lower().endswith((".png", ".jpg", ".jpeg", "decal.png")) and not p.lower().endswith(
            ("_n.png", "_g.png", "environmentmask.png"))
    ]

    if not image_paths:
        print(f"No images found recursively in {input_base_dir}")
        return

    print(f"Found {len(image_paths)} images for processing.")

    # 3. Process all found images
    for hr_img_path in image_paths:
        # Determine the relative path from the input base directory
        # e.g., if input_base_dir is '/data/input' and hr_img_path is
        # '/data/input/subdir/image.png', relative_path will be 'subdir/image.png'
        relative_path = os.path.relpath(hr_img_path, input_base_dir)

        # Construct the final save path
        # e.g., '/data/output/subdir/image.png'
        final_save_path = os.path.join(output_base_dir, relative_path)

        # 4. Perform patch-based inference
        print("-" * 40)
        print(f"Processing: {relative_path}")
        print(f"Saving to: {final_save_path}")

        # The output tensor is returned but not used in the original loop.
        _ = gan_inference_patch_based_rec(
            model,
            hr_img_path,
            patch_size=CONFIG['resolution'],
            scale=CONFIG['scale'],
            device=device,
            # Pass the full target save path
            results_dir=final_save_path
        )

    print("\nProcessing complete for all images.")


if __name__ == "__main__":
    # Change the main execution block to include your new test function
    test = True
    if test:
        gan_test_rec()
    else:
        train()
