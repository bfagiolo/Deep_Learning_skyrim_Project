import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
import numpy as np
import argparse
import re, glob, os



# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, help="the data key of the set. EX: ARCHIT for architecture textures.")
args = parser.parse_args()


# Test dataset preparation
def prepare_test_dataset(TEXTURE_DIR, PATCH_SIZE, BATCH_SIZE):
    """Create test dataset from the texture directory"""
    import glob
    import random
    
    # Get all paths
    all_paths = glob.glob(os.path.join(TEXTURE_DIR, "*"))
    all_paths = [p for p in all_paths 
                 if os.path.isfile(p) 
                 and p.lower().endswith(('.png', '.jpg', '.jpeg', '.dds'))]
    
    # CRITICAL: Use same seed as training to get same test set
    random.seed(42)
    random.shuffle(all_paths)
    random.seed(None)
    
    # Same split as training
    total = len(all_paths)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_paths = all_paths[train_size + val_size:]
    
    print(f"Test set: {len(test_paths)} images from {TEXTURE_DIR}")
    
    # Import here to avoid circular dependency
    from DL_train import PatchTextureDataset
    
    test_ds = PatchTextureDataset(
        root_dir=TEXTURE_DIR,
        patch_size=PATCH_SIZE,
        patches_per_image=1,  # Use 1 patch per image for testing
        paths=test_paths,
        cache_in_memory=False  # Don't cache for testing
    )
    
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                     num_workers=4, drop_last=False, pin_memory=True)

# Calculate PSNR
def calculate_psnr(img1, img2):
    """Calculate PSNR between two images in [-1, 1] range"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    # For [-1, 1] range, max pixel value is 2
    psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)
    return psnr.item()

# Calculate SSIM for a batch
def calculate_ssim_batch(img1_batch, img2_batch):
    """Calculate SSIM for a batch of images"""
    # Convert from [-1, 1] to [0, 1]
    img1_batch = (img1_batch + 1) / 2
    img2_batch = (img2_batch + 1) / 2
    
    # Move to CPU and convert to numpy
    img1_np = img1_batch.cpu().detach().numpy()
    img2_np = img2_batch.cpu().detach().numpy()
    
    ssim_scores = []
    for i in range(img1_np.shape[0]):
        # SSIM expects (H, W, C) format
        img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
        img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
        
        score = ssim(img1_hwc, img2_hwc, 
                    data_range=1.0,
                    channel_axis=2)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

# Calculate metrics
def calculate_metrics(model, dataloader, lpips_fn, DEVICE, MAX_TRAINING_NOISE_LEVEL,add_noise_level,get_timestep_from_noise_level,sample_from_partial, cosine_beta_schedule,TIMESTEPS,RESULTS_DIR ):
    """Calculate PSNR, SSIM, and LPIPS metrics"""
    psnr_list = []
    ssim_list = []
    lpips_list = []
    
    with torch.no_grad():
        for batch_idx, x0 in enumerate(tqdm(dataloader, desc="Testing")):
            x0 = x0.to(DEVICE)
            
            # Add noise to the original image
            xk_batch = add_noise_level(x0, MAX_TRAINING_NOISE_LEVEL)
            
            # Get timestep
            betas = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
            alphas = 1.0 - betas
            alpha_hat = torch.cumprod(alphas, dim=0)

            k = get_timestep_from_noise_level(
                torch.tensor(MAX_TRAINING_NOISE_LEVEL).to(DEVICE), 
                alpha_hat
            )
            
            # Denoise
            print(type(model))
            x_denoised = sample_from_partial(model, xk_batch, k)
            
            # Calculate PSNR for each image in batch
            for i in range(x0.shape[0]):
                psnr = calculate_psnr(x_denoised[i:i+1], x0[i:i+1])
                psnr_list.append(psnr)
            
            # Calculate SSIM for batch
            ssim_value = calculate_ssim_batch(x_denoised, x0)
            ssim_list.append(ssim_value)
            
            # Calculate LPIPS (expects [-1, 1] range)
            lpips_value = lpips_fn(x_denoised, x0)
            lpips_list.append(lpips_value.mean().item())
            
            # Save comparison images for first few batches
            if batch_idx < 5:
                save_comparisons(x0, xk_batch, x_denoised, batch_idx,RESULTS_DIR)
    
    # Return average metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    
    return avg_psnr, avg_ssim, avg_lpips

# Save comparison images
def save_comparisons(x0_batch, xk_batch, x_denoised, batch_idx, RESULTS_DIR):
    """Save side-by-side comparison of ground truth, degraded, and reconstructed"""
    all_imgs = torch.cat([
        (x0_batch + 1) / 2,      # Ground truth (clean)
        (xk_batch + 1) / 2,      # Degraded input (noisy)
        (x_denoised + 1) / 2     # Reconstruction (denoised)
    ], dim=0)
    
    save_path = os.path.join(RESULTS_DIR, f"comparison_batch_{batch_idx}.png")
    save_image(all_imgs, save_path, nrow=x0_batch.shape[0])
    
    if batch_idx == 0:
        print(f"\nSaved comparison to: {save_path}")
        print("  Row 1: Ground Truth")
        print("  Row 2: Degraded Input")
        print("  Row 3: Reconstruction")

# Main function for testing
def test(model,TEXTURE_DIR, PATCH_SIZE, BATCH_SIZE,DEVICE,MODEL_DIR,MAX_TRAINING_NOISE_LEVEL,add_noise_level,get_timestep_from_noise_level,sample_from_partial,  cosine_beta_schedule,TIMESTEPS,RESULTS_DIR):
    print("=" * 60)
    print("TEXTURE DIFFUSION MODEL - TESTING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {MODEL_DIR}")
    print(f"Noise Level: {MAX_TRAINING_NOISE_LEVEL}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model not found at {MODEL_DIR}")
        print("Please train the model first!")
        return
    
    # Initialize LPIPS
    print("Initializing LPIPS metric...")
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)  # Using AlexNet
    
    # Prepare the test dataset
    print("Preparing test dataset...")
    test_dl = prepare_test_dataset(TEXTURE_DIR, PATCH_SIZE, BATCH_SIZE)
    
    if len(test_dl.dataset) == 0:
        print("ERROR: No test images found!")
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    avg_psnr, avg_ssim, avg_lpips = calculate_metrics(model, test_dl, lpips_fn, DEVICE, MAX_TRAINING_NOISE_LEVEL,add_noise_level,get_timestep_from_noise_level,sample_from_partial,  cosine_beta_schedule,TIMESTEPS,RESULTS_DIR)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Average PSNR:  {avg_psnr:.4f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print("=" * 60)
    print(f"\nComparison images saved to: {RESULTS_DIR}")
    print("Testing completed!")

# if __name__ == "__main__":
#     test()