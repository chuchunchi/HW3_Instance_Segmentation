"""
Dataset module for cell instance segmentation.

This module provides datasets for training and inference of cell instance segmentation models.
It handles loading and preprocessing of TIFF images and their corresponding masks.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from skimage import measure
from typing import Dict, List, Optional, Tuple, Union

__all__ = ["CellDataset", "TestImageDataset", "collate_fn"]


class CellDataset(Dataset):
    """Dataset for histopathology cell instance segmentation.
    
    This dataset loads TIFF images and their corresponding instance masks for training.
    It prepares instance-level masks and bounding boxes compatible with Torchvision's Mask R-CNN.
    Degenerate (zero-width/zero-height) bounding boxes are filtered out to comply with the model's requirements.
    
    Args:
        root_dir: Path to the root directory containing sample folders
        transforms: Optional transform to be applied on the image and target
    """
    
    # Mapping of class IDs to their corresponding mask filenames
    CLASS_FILENAMES: Dict[int, str] = {
        1: "class1.tif",
        2: "class2.tif",
        3: "class3.tif",
        4: "class4.tif"
    }

    def __init__(self, root_dir: Union[str, Path], transforms: Optional[callable] = None):
        """Initialize the dataset.
        
        Args:
            root_dir: Path to the root directory containing sample folders
            transforms: Optional transform to be applied on the image and target
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.sample_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        
        if not self.sample_dirs:
            raise RuntimeError(f"No sample folders found in {self.root_dir}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing:
                - image: Tensor of shape (3, H, W) containing the RGB image
                - target: Dictionary containing:
                    - boxes: Tensor of shape (N, 4) containing bounding boxes
                    - labels: Tensor of shape (N,) containing class labels
                    - masks: Tensor of shape (N, H, W) containing instance masks
                    - image_id: Tensor containing the image ID
                    - area: Tensor of shape (N,) containing box areas
                    - iscrowd: Tensor of shape (N,) containing crowd flags
        """
        sample_dir = self.sample_dirs[idx]

        # Load RGB image
        image = self._read_rgb(sample_dir / "image.tif")

        # Load and process instance masks
        raw_masks: List[np.ndarray] = []
        raw_labels: List[int] = []
        
        for class_id, filename in self.CLASS_FILENAMES.items():
            mask_path = sample_dir / filename
            if not mask_path.exists():
                continue
                
            # Load and binarize mask
            binary_mask = (tifffile.imread(str(mask_path)) > 0).astype(np.uint8)
            
            # Find connected components
            connected_components = measure.label(binary_mask, connectivity=1)
            
            # Extract individual instance masks
            for instance_id in range(1, connected_components.max() + 1):
                instance_mask = connected_components == instance_id
                if instance_mask.sum() == 0:
                    continue
                raw_masks.append(instance_mask)
                raw_labels.append(class_id)

        if not raw_masks:
            raise RuntimeError(f"No instances found in {sample_dir}")

        # Process bounding boxes and filter degenerate instances
        boxes: List[List[float]] = []
        valid_masks: List[np.ndarray] = []
        valid_labels: List[int] = []
        
        for mask, label in zip(raw_masks, raw_labels):
            y_coords, x_coords = np.where(mask)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()
            
            # Filter out degenerate boxes
            if x_max > x_min and y_max > y_min and x_min != 0 and y_min != 0:
                boxes.append([x_min, y_min, x_max, y_max])
                valid_masks.append(mask)
                valid_labels.append(label)
                
        if not boxes:
            raise RuntimeError(f"All instances in {sample_dir} are degenerate (size=0).")

        # Convert to tensors
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        masks = torch.as_tensor(np.stack(valid_masks, axis=0), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        """Read and preprocess an RGB image from a TIFF file.
        
        Args:
            path: Path to the TIFF file
            
        Returns:
            Numpy array of shape (3, H, W) containing the RGB image
        """
        img = tifffile.imread(str(path))
        
        # Handle different input formats
        if img.ndim == 2:  # Grayscale
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:  # RGBA or multi-channel
            img = img[:, :, :3]
            
        # Convert to C x H x W format
        return img.transpose(2, 0, 1)


class TestImageDataset(Dataset):
    """Dataset for inference on test images.
    
    This dataset loads test images for inference without requiring masks or labels.
    
    Args:
        root: Path to the directory containing test images
    """
    
    def __init__(self, root: Union[str, Path]):
        """Initialize the test dataset.
        
        Args:
            root: Path to the directory containing test images
        """
        self.root = Path(root)
        self.image_files = sorted(list(self.root.glob("*.tif")))
        self.image_paths = [str(p) for p in self.image_files]

    def __len__(self) -> int:
        """Return the number of test images."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get a test image.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            Tuple containing:
                - image: Tensor of shape (3, H, W) containing the RGB image
                - image_path: Path to the image file
        """
        path = self.image_files[idx]
        image = tifffile.imread(str(path))
        
        # Convert to RGB format
        if image.ndim == 2:  # Grayscale
            image = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[2] == 3:  # RGB
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:  # Multi-channel
            image = torch.from_numpy(image[:, :, :3]).permute(2, 0, 1).float()
            
        # Normalize to [0, 1]
        image /= 255.0
        
        return image, self.image_paths[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], str]]]) -> Tuple[List[torch.Tensor], List[Union[Dict[str, torch.Tensor], str]]]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of tuples containing images and their targets/paths
        
    Returns:
        Tuple containing:
            - List of images
            - List of targets/paths
    """
    return tuple(zip(*batch))
