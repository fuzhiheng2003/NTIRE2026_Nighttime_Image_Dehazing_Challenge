# TripleSpaceDehazeNet
#  TeamName: JNU620

Overview
--------
TripleSpaceDehazeNet is JNU620 submission for NTIRE2026-Nighttime Image Dehazing Challenge. 
We propose TripleSpaceDehazeNet, a nighttime image dehazing network built on two complementary design principles.
First, inspired by SGDN, we extend the dual color-space guidance strategy from RGB+YCbCr to a triple-space representation that additionally incorporates the CIE Lab color space.
Second, motivated by the multi-prior guidance framework in ClearNight, we introduce a joint physical prior map that simultaneously encodes dark channel statistics, image gradient structure, illumination distribution, and local smoothness—four cues that together capture the heterogeneous nature of nighttime haze.

Repository layout
-----------------
- `data/` -The 'data' folder is used to store dataset images.
- `results` -The 'results' folder is used to store the final images submitted for the competition.
- `weights/` -The 'weight' folder stores the model weights obtained from the final training of the competition, used to quickly reproduce the final submission results by running the infer.py inference script.
- `checkpoints/` -The checkpoints folder is used to store the weights produced by model training.
- `train_results/` -The train_results folder is used to store the visual dehazed images generated during the model training process.


Pretrained weights
------------------
Place the trained checkpoint here:
```
checkpoints/model_latest.pth
```

Environment
-----------
Recommended setup (same as repository root):

```bash
conda create -n dehaze python=3.9 -y
conda activate dehaze
pip install -r requirements.txt
```

Quick test (repository root)
----------------------------
```bash
python infer.py
```
After running the infer.py script, the dehazed images processed by the model will be output to the output_submission folder

Quick train (repository root)
---------------------------- 
```bash
python train.py
```
You can start training by running the training script directly in the project's root directory. After training starts, two additional folders will appear: the 'checkpoints' folder contains the model's training weights, and the 'train_result' folder contains visualized dehazed images from the training process.
If you need to test the retrained model weights, you need to change the address pointing to the retrained weights in the infer.py file, and then run it.


