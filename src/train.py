"""train.py – Training script with Albumentations augmentation, 10% val split, loss plotting, and fixed evaluation losses."""
from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import CellDataset, collate_fn


def get_train_transform():
    """Get training data augmentation transforms."""
    albu = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
        A.Rotate(limit=20, p=0.3),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        A.OneOf([A.GaussNoise(), A.GaussianBlur()], p=0.15),
        ToTensorV2(transpose_mask=True),
    ], additional_targets={"masks": "masks"})
    return AlbuAdapter(albu)


def get_val_transform():
    """Get validation data transforms."""
    return AlbuAdapter(A.Compose([ToTensorV2(transpose_mask=True)]))


class AlbuAdapter:
    """Adapter for Albumentations transforms to work with PyTorch tensors."""
    def __init__(self, albu_transform):
        self.t = albu_transform

    def __call__(self, image: torch.Tensor, target: dict):
        img_np = (image.mul(255).permute(1, 2, 0).byte().cpu().numpy())
        masks_np = target['masks'].cpu().numpy()
        labels_np = target['labels'].cpu().numpy()
        
        aug = self.t(image=img_np, masks=list(masks_np))
        img = aug['image'].float() / 255.0
        
        new_m, new_b, new_l = [], [], []
        for m, l in zip(aug['masks'], labels_np):
            ys, xs = np.where(m)
            if ys.size == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            if x1 <= x0 or y1 <= y0:
                continue
            new_m.append(m)
            new_b.append([x0, y0, x1, y1])
            new_l.append(l)
            
        if not new_m:
            return image, target
            
        masks = torch.as_tensor(np.stack(new_m), dtype=torch.uint8)
        boxes = torch.as_tensor(new_b, dtype=torch.float32)
        labels = torch.as_tensor(new_l, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)
        
        return img, {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': target['image_id'],
            'area': area,
            'iscrowd': iscrowd
        }


def get_model(num_classes: int, model: str):
    """Get Mask R-CNN model with custom number of classes."""
    
    if model == 'v2':
        # Create model without FPN
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            backbone_kwargs={"norm_layer": torch.nn.BatchNorm2d}
        )
    else:
        # Create model with FPN (default)
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT",
            backbone_kwargs={"norm_layer": torch.nn.BatchNorm2d}
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, targets in dataloader:
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.inference_mode()
def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    was_train = model.training
    model.train()  # Force training mode to compute loss
    total_loss = 0.0
    
    for images, targets in dataloader:
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    
    if not was_train:
        model.eval()
    
    return total_loss / len(dataloader)


def main(cfg):
    """Main training function."""
    # Prepare datasets
    train_ds = CellDataset(cfg.data_root, transforms=get_train_transform())
    val_ds = CellDataset(cfg.data_root, transforms=get_val_transform())
    
    # Split into train and validation
    indices = list(range(len(train_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    
    # Create dataloaders
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg.batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=cfg.batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(5, cfg.model).to(device)
    
    # Setup optimizer based on choice
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Setup checkpointing
    out_dir = Path(cfg.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = out_dir / 'best_model_adamw.pth'
    
    # Load checkpoint if exists
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{cfg.epochs} train={train_loss:.4f} val={val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
    
    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.savefig(out_dir / 'loss_curve.png')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/train')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--outdir', default='output')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd',
                      help='Choose optimizer: sgd or adamw')
    parser.add_argument('--model', choices=['fpn', 'v2'], default='fpn',
                      help='Choose model: fpn or v2')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
