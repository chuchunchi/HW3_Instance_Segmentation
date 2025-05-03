# HW3_Instance_Segmentation
Homework 3 of VRDL class, Instance Segmentation Task

Student ID: 313551057

## Requirements

Install Install python 3.11 and the required packages:
```bash
cd HW3_Instance_Segmentation
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── train.py          # Training script
│   ├── inference.py      # Inference script
│   ├── visualization.py  # Debug and visualize predictions
│   └── dataset.py        # Dataset handling
├── data/
│   ├── train/            # Training data
│   └── test/             # Test data
├── output/                 # Output directory for training results
└── test_image_name_to_ids.json # image name to id mapping
```

## Scripts

### 1. Training (`train.py`)

Train a Mask R-CNN model with different configurations.

**Usage:**
```bash
python src/train.py \
    --data_root data/train \
    --epochs 25 \
    --batch 2 \
    --lr 5e-3 \
    --outdir output \
    --optimizer adamw \
    --model fpn
```

**Arguments:**
- `--data_root`: Path to training data directory
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--lr`: Learning rate
- `--outdir`: Output directory for checkpoints and logs
- `--optimizer`: Optimizer choice ('sgd' or 'adamw')
- `--model`: Model type ('fpn' or 'v2')

**Features:**
- Supports both FPN and FPN v2 backbones
- Implements SGD and AdamW optimizers
- Includes learning rate scheduling
- Saves best model based on validation loss
- Generates loss curves

### 2. Inference (`inference.py`)

Run inference on test images and generate results in COCO format.

**Usage:**
```bash
python src/inference.py \
    --data_root data/test \
    --ckpt output/best_model.pth \
    --output results.json \
    --model fpn
```

**Arguments:**
- `--data_root`: Path to test data directory
- `--ckpt`: Path to model checkpoint
- `--output`: Output JSON file for results
- `--model`: Model type ('fpn' or 'v2')

**Features:**
- Supports both FPN and FPN v2 models
- Generates COCO-format predictions
- Includes bounding boxes, masks, and confidence scores
- Saves results in JSON format

### 3. Debug Predictions (`visualization.py`)

Visualize and debug model predictions on test images.

**Usage:**
```bash
python src/visualization.py \
    --data_root data/test \
    --ckpt runs/experiment_name/best_model.pth \
    --output debug/
```

**Arguments:**
- `--data_root`: Path to test data directory
- `--ckpt`: Path to model checkpoint
- `--output`: Output directory for visualizations

**Features:**
- Visualizes predictions with segmentation masks
- Color-codes different cell classes
- Shows confidence scores
- Saves visualizations as PNG files
- Generates prediction statistics


## Output Format

### Training Output
- Checkpoints saved in `output`
- Loss curves saved as `loss_curve.png`
- Best model saved as `best_model.pth`

### Inference Output
- JSON file containing predictions in COCO format
- Each prediction includes:
  - Image ID
  - Category ID
  - Bounding box
  - Confidence score
  - Segmentation mask (RLE format)

### Debug Output
- Visualizations saved as PNG files
- Prediction statistics printed to console
- Raw prediction data saved as JSON files 