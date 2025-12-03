# Deep_Learning_skyrim_Project
Deep Learning Final Project: This project uses a hybrid texture-restoration pipeline that combines the strengths of GANs and diffusion. A GAN first learns to upscale and add detail to texture patches, capturing the sharp, high-frequency structure that GANs excel at. A diffusion model then separately learns to denoise and refine those patches with more stable, artifact-free results. During inference, textures are split into 256×256 tiles, enhanced by the GAN, cleaned up by diffusion, and stitched back into full-resolution images. The result is a lightweight, VRAM-friendly system that produces sharp, clean, high-quality textures for large game assets like Skyrim.


How to train on a VM:

Run the vm by clicking ssh
Install gcloud
https://docs.cloud.google.com/sdk/docs/install-sdk
Run gcloud init
Log into your cornell gmail
Install fuse:
sudo apt-get update
sudo apt-get install -y curl lsb-release
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse
Run mkdir "$HOME/data"
Download the data:
gcloud storage cp --recursive gs://skyrim-inference-data/ /demo
/workspace
git clone https://github.com/bfagiolo/Deep_Learning_skyrim_Project.git
Cd into the folder containing your .py and Edit your data path:
vi DL_train.py
If training on our dataset:
In the main method set _type to “train”
run DL_train.py -d [TEXTURE_KEY] (texture keys are defined at the top of the file)
If training on your own dataset:
Set TEXTURE_DIR ‘s default value to your data path
run DL_train.py
If conducting inference:
set _type to “test”
Edit any parameters such as MAX_TRAINING_NOISE_LEVEL
run the file with python DL_train.py

Installing cuda drivers:
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network

Bullseye backport install error:
https://docs.google.com/document/d/1TeAL4X5zR0RfB-tISUzvWl4zYyI4I_PgXzcB37aZYL4/edit?tab=t.0
it formatted like this. make adjustments so its formated cleanly

Attachment
Screenshot-2025-12-02-at-9.54.36-PM.jpg
text
## How to train our Diffusion on a VM

1. Run the VM by clicking **SSH**.

2. Install gcloud:  
   https://docs.cloud.google.com/sdk/docs/install-sdk  

3. Run `gcloud init`.

4. Log into your Cornell Gmail.

5. Install fuse:
sudo apt-get update
sudo apt-get install -y curl lsb-release
export GCSFUSE_REPO=gcsfuse-lsb_release -c -s
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse

text

6. Create a data directory:
mkdir "$HOME/data"

text

7. Download the data:
gcloud storage cp --recursive gs://skyrim-inference-data/ /demo

text

8. Clone the repo in `/workspace`:
cd /workspace
git clone https://github.com/bfagiolo/Deep_Learning_skyrim_Project.git

text

9. Cd into the folder containing your `.py` and edit your data path:
cd Deep_Learning_skyrim_Project
vi DL_train.py

text

### If training on our dataset
- In the `main` method set `_type` to `"train"`.  
- Run:
python DL_train.py -d [TEXTURE_KEY]

text
(texture keys are defined at the top of the file)

### If training on your own dataset
- Set `TEXTURE_DIR`’s default value to your data path.  
- Run:
python DL_train.py

text

### If conducting inference
- Set `_type` to `"test"`.  
- Edit any parameters such as `MAX_TRAINING_NOISE_LEVEL`.  
- Run:
python DL_train.py

text

---

## Installing CUDA drivers

CUDA 12.4 for Debian 11 (x86_64):  
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network  

## Bullseye backport install error

Debug notes:  
https://docs.google.com/document/d/1TeAL4X5zR0RfB-tISUzvWl4zYyI4I_PgXzcB37aZYL4/edit?tab=t.0
