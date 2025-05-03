import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks
import tifffile

from inference import load_model
from dataset import TestImageDataset


def tensor_to_list(tensor):
    """Convert tensor to list for JSON serialization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    return tensor


def visualize_predictions(image, predictions, output_path):
    """Visualize predictions on an image.
    
    Args:
        image: Input image tensor of shape (C, H, W)
        predictions: Dictionary containing model predictions
        output_path: Path to save the visualization
    """
    # Convert image to RGB if grayscale
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    # Define consistent colors for each class
    class_colors = {
        1: 'red',      # Class 1
        2: 'green',    # Class 2
        3: 'blue',     # Class 3
        4: 'yellow',   # Class 4
        5: 'cyan'      # Class 5
    }
    
    # Get predictions with score > 0.5
    high_conf_idx = predictions['scores'] > 0.5
    if high_conf_idx.any():
        masks = predictions['masks'][high_conf_idx, 0] > 0.5
        scores = predictions['scores'][high_conf_idx]
        class_ids = predictions['labels'][high_conf_idx]
        
        # Sort by class ID and then by score for consistent visualization
        sorted_idx = torch.argsort(class_ids * 1000 + scores, descending=False)
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]
        class_ids = class_ids[sorted_idx]
        
        # Get colors for each mask based on class
        colors = [class_colors[class_id.item()] for class_id in class_ids]
        
        # Create labels for each detection
        labels = []
        for score, class_id in zip(scores, class_ids):
            labels.append(f"Class {class_id.item()} ({score.item():.2f})")
        
        # Draw masks on image
        result = draw_segmentation_masks(
            (image * 255).byte(),
            masks,
            alpha=0.4,  # Slightly more transparent for better visibility
            colors=colors
        )
        
        # Save visualization
        plt.figure(figsize=(15, 15))
        plt.imshow(result.permute(1, 2, 0).numpy())
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization with {len(masks)} detections")
    else:
        print("No high confidence detections to visualize")


def debug_predictions(args):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = load_model(Path(args.ckpt), num_classes=5, device=device)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load test dataset
    dataset = TestImageDataset(args.data_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Process each image
    for i, (images, image_ids) in enumerate(dataloader):
        if i >= 10:  # Only process first 5 images for debugging
            break
            
        images = [img.to(device) for img in images]
        with torch.no_grad():
            predictions = model(images)
        
        # Convert predictions to JSON-serializable format
        pred_dict = {
            'boxes': tensor_to_list(predictions[0]['boxes']),
            'labels': tensor_to_list(predictions[0]['labels']),
            'scores': tensor_to_list(predictions[0]['scores']),
            'masks': tensor_to_list(predictions[0]['masks'])
        }
        
        # Save raw predictions
        with open(output_dir / f'predictions_{i}.json', 'w') as f:
            json.dump(pred_dict, f, indent=2)
        
        # Print prediction statistics
        print(f"\nImage {i+1} Predictions:")
        print(f"Number of detections: {len(predictions[0]['boxes'])}")
        print(f"Class distribution: {np.bincount(predictions[0]['labels'].cpu().numpy())}")
        print(f"Score range: [{predictions[0]['scores'].min():.3f}, {predictions[0]['scores'].max():.3f}]")
        
        # Visualize predictions
        visualize_predictions(
            images[0].cpu(),
            predictions[0],
            output_dir / f'visualization_{i}.png'
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Debug model predictions')
    parser.add_argument('--data_root', type=str, required=True, help='Path to test images')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='debug', help='Output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    debug_predictions(args) 