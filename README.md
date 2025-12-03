# How to train our Diffusion on a VM:

Run the vm by clicking ssh
Install gcloud
[https://docs.cloud.google.com/sdk/docs/install-sdk](https://docs.cloud.google.com/sdk/docs/install-sdk)

Run `gcloud init`
Log into your cornell gmail

Install fuse:

```
sudo apt-get update
sudo apt-get install -y curl lsb-release
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update
sudo apt-get install gcsfuse
```

Run `mkdir "$HOME/data"`

Download the data:

```
gcloud storage cp --recursive gs://skyrim-inference-data/ /demo
```

```
/workspace
git clone https://github.com/bfagiolo/Deep_Learning_skyrim_Project.git
```

Cd into the folder containing your .py and Edit your data path:

```
vi DL_train.py
```

---

## If training on our dataset:

In the main method set `_type` to “train”
run:

```
DL_train.py -d [TEXTURE_KEY]
```

(texture keys are defined at the top of the file)

---

## If training on your own dataset:

Set `TEXTURE_DIR`'s default value to your data path
run:

```
DL_train.py
```

---

## If conducting inference:

set `_type` to “test”
run:

```
DL_train.py [Input directory] [output directory]
```

this recursively walks through input dir and any subfolders, and copies it to output directory while preserving the same structure.
The image path saving may have some bugs, so when running final inference I just set it to overwrite the original (aka input dir = output dir)
I may have hard coded some paths in testing, so if it does not find the path you may need to look for long string literals in the infer pipeline

Edit any parameters such as `MAX_TRAINING_NOISE_LEVEL`
run the file with:

```
python DL_train.py
```

---

# Installing cuda drivers:

[https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network)

---

# Bullseye backport install error:

[https://docs.google.com/document/d/1TeAL4X5zR0RfB-tISUzvWl4zYyI4I_PgXzcB37aZYL4/edit?tab=t.0](https://docs.google.com/document/d/1TeAL4X5zR0RfB-tISUzvWl4zYyI4I_PgXzcB37aZYL4/edit?tab=t.0)

---

Let me know if you want this turned into a PDF or a README.md file.
