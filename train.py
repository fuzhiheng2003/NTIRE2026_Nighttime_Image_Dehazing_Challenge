import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import random
import cv2
from copy import deepcopy

from dataset import DehazeDataset
from model import TripleSpaceDehazeNet
from loss import TripleLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
EPOCHS = 2000
LR = 2e-4*BATCH_SIZE
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-4
TRAIN_DATA_PATH = 'data/train'
SEED = 3407

def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.device = next(model.parameters()).device
        self.model.to(self.device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            emsd = self.model.state_dict()
            for name, param in msd.items():
                if param.dtype in [torch.float16, torch.float32]:
                    emsd[name].copy_(self.decay * emsd[name] + (1. - self.decay) * param)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def visualize_progress(model, epoch, device, save_dir='training_results'):
    model.eval()
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    test_img_path = 'data/test/31_NTHazy.png'
    if not os.path.exists(test_img_path):
        possible_dir = os.path.join(TRAIN_DATA_PATH, 'hazy')
        if os.path.exists(possible_dir):
            files = [f for f in os.listdir(possible_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if files: test_img_path = os.path.join(possible_dir, files[0])
        else:
            return

    img = cv2.imread(test_img_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    if new_h != h or new_w != w:
        img = cv2.resize(img, (new_w, new_h))

    img_tensor = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        with autocast():
            output = model(img_tensor)
            output = output.float()
    torch.cuda.empty_cache()
    gc.collect()
    output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f'epoch_{epoch + 1}_visual.png'), output_img)
    model.train()


def train():
    seed_everything(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = DehazeDataset(TRAIN_DATA_PATH, mode='train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    print(f"Model initializing... Device: {DEVICE}")
    model = TripleSpaceDehazeNet().to(DEVICE)
    ema = ModelEMA(model, decay=0.999)

    criterion = TripleLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=1, eta_min=MIN_LR)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"Start training | Epochs: {EPOCHS} | Seed: {SEED}")
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=False)

        for hazy, clear in loop:
            hazy = hazy.to(DEVICE)
            clear = clear.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):

                output = model(hazy)
                loss = criterion(output, clear)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            visualize_progress(ema.model, epoch, DEVICE)

        torch.save(ema.model.state_dict(), 'checkpoints/model_latest.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }, 'checkpoints/training_state_latest.pth')

        if (epoch + 1) >= (EPOCHS * 0.6):
            if (epoch + 1) % 20 == 0:
                save_path = f'checkpoints/model_epoch_{epoch + 1}.pth'
                torch.save(ema.model.state_dict(), save_path)

if __name__ == '__main__':
    train()