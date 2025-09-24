"""
Prepare dataset by splitting a single directory of images into train/val/test splits.

Usage:
    python prepare_dataset.py --input_dir raw_images --output_dir dataset

Expected input structure:
    raw_images/
        A/
            *.jpg
        B/
            *.jpg
        ...
        SPACE/
        DELETE/
        NOTHING/

Output structure:
    dataset/
        train/
            A/ (70% of images)
            B/
            ...
        val/
            A/ (15% of images)
            B/
            ...  
        test/
            A/ (15% of images)
            B/
            ...
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test maintaining class distribution"""
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {input_dir}")
    
    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    total_images = 0
    split_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            logger.warning(f"No images found in {class_dir}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate splits
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        n_test = n_images - n_train - n_val  # Remaining images go to test
        
        logger.info(f"{class_name}: {n_images} images -> train: {n_train}, val: {n_val}, test: {n_test}")
        
        # Create class directories in output
        for split in ['train', 'val', 'test']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Split and copy files
        splits = [
            ('train', image_files[:n_train]),
            ('val', image_files[n_train:n_train + n_val]),
            ('test', image_files[n_train + n_val:])
        ]
        
        for split_name, files in splits:
            split_counts[split_name][class_name] = len(files)
            
            for img_file in files:
                dest_path = output_path / split_name / class_name / img_file.name
                shutil.copy2(img_file, dest_path)
        
        total_images += n_images
    
    # Print summary
    logger.info(f"\nDataset split completed!")
    logger.info(f"Total images processed: {total_images}")
    
    for split_name in ['train', 'val', 'test']:
        total_split = sum(split_counts[split_name].values())
        logger.info(f"\n{split_name.upper()} set: {total_split} images")
        for class_name, count in sorted(split_counts[split_name].items()):
            percentage = (count / total_split) * 100 if total_split > 0 else 0
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Save split information
    split_info = {
        'total_images': total_images,
        'splits': dict(split_counts),
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'seed': seed
    }
    
    import json
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"\nSplit information saved to: {output_path / 'split_info.json'}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset with train/val/test splits')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing class subdirectories')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of images for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of images for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of images for testing (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            args.input_dir,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        print(f"\n✅ Dataset preparation completed successfully!")
        print(f"Dataset ready for training at: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

if __name__ == '__main__':
    main()