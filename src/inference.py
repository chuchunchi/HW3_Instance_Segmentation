import argparse
import json
from pathlib import Path
import tifffile
import numpy as np
import pycocotools.mask as mask_util

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dataset import TestImageDataset


def load_model(ckpt_path: Path, num_classes: int, device: torch.device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat_box, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_image_ids(json_path: Path) -> dict:
    """Load image ID mapping from JSON file."""
    with open(json_path) as f:
        id_map = json.load(f)
    return {item['file_name']: item['id'] for item in id_map}


def run_inference(args, device):
    # Load image ID mapping
    id_map = load_image_ids(Path('test_image_name_to_ids.json'))
    
    ds = TestImageDataset(args.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = load_model(Path(args.ckpt), num_classes=5, device=device)
    output = []
    
    for images, image_paths in loader:
        images = [i.to(device) for i in images]
        with torch.no_grad():
            outs = model(images)
            
        for out, img_path in zip(outs, image_paths):
            # Get image ID from filename
            img_name = Path(img_path).name
            img_id = id_map.get(img_name)
            if img_id is None:
                print(f"Warning: No ID found for image {img_name}")
                continue
                
            boxes = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            labels = out['labels'].cpu().numpy()
            masks = (out['masks'].cpu().numpy() > 0.5).astype(np.uint8)
            
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                # Convert box from [x1,y1,x2,y2] to [x1,y1,width,height]
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Convert mask to RLE format
                rle = mask_util.encode(np.asfortranarray(mask[0]))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                output.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(score),
                    'segmentation': {
                        'size': [int(rle['size'][0]), int(rle['size'][1])],
                        'counts': rle['counts']
                    }
                })
    
    with open(args.output, 'w') as f:
        json.dump(output, f)
    print(f"Saved inference results to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--output', type=str, default='results.json')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_inference(args, device)


if __name__ == '__main__':
    main()
