# TripleSpaceDehazeNet
#  TeamName: JNU620
Overview
--------
TripleSpaceDehazeNet is JNU620 submission for NTIRE2026-Nighttime Image Dehazing Challenge. 

Repository layout
-----------------
data

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


