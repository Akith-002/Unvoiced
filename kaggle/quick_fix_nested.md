# QUICK FIX FOR YOUR KAGGLE NOTEBOOK

# Replace your Cell 2 with this:

## Cell 2: Find Dataset (FIXED for nested structure)

```python
from pathlib import Path

# Your dataset has double nesting - fix the path
dataset_path = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'

print(f"🔍 Using dataset path: {dataset_path}")

# Verify the structure
dataset_check = Path(dataset_path)
if dataset_check.exists():
    print(f"✅ Dataset found!")

    # Show contents
    folders = [d for d in dataset_check.iterdir() if d.is_dir()]
    print(f"📁 Found {len(folders)} folders:")

    for folder in sorted(folders)[:10]:  # Show first 10
        img_count = len(list(folder.glob('*.jpg')) + list(folder.glob('*.png')))
        print(f"  {folder.name}: {img_count} images")

    if len(folders) > 10:
        print(f"  ... and {len(folders) - 10} more folders")

    print(f"\n🎯 Total folders: {len(folders)}")

else:
    print("❌ Dataset path not found!")
```

## Cell 3: Prepare Dataset (FIXED)

```python
def prepare_dataset(input_dir, output_dir):
    print(f"📁 Preparing dataset from: {input_dir}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes: {sorted([d.name for d in class_dirs])}")

    if len(class_dirs) == 0:
        print("❌ No class folders found!")
        return None

    splits = ['train', 'val', 'test']
    split_ratios = [0.7, 0.15, 0.15]

    for split in splits:
        (output_path / split).mkdir(exist_ok=True)

    total_files = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for class_dir in class_dirs:
        print(f"Processing: {class_dir.name}")

        # Look for images
        image_files = (list(class_dir.glob('*.jpg')) +
                      list(class_dir.glob('*.png')) +
                      list(class_dir.glob('*.jpeg')))
        image_files.sort()

        if len(image_files) == 0:
            print(f"  ⚠️  No images in {class_dir.name}")
            continue

        print(f"  Found {len(image_files)} images")
        total_files += len(image_files)

        # Split the images
        n_files = len(image_files)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])

        # Create class directories in each split
        for split in splits:
            (output_path / split / class_dir.name).mkdir(exist_ok=True)

        # Copy files to splits
        for i, img_file in enumerate(image_files):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'

            dst = output_path / split / class_dir.name / img_file.name
            shutil.copy2(img_file, dst)
            split_counts[split] += 1

    print(f"✅ Dataset prepared: {total_files:,} total files")
    print(f"   Train: {split_counts['train']:,}")
    print(f"   Val: {split_counts['val']:,}")
    print(f"   Test: {split_counts['test']:,}")
    return str(output_path)

# Use the CORRECT nested path
prepared_dataset = prepare_dataset(dataset_path, '/kaggle/working/dataset')

if prepared_dataset:
    print(f"🎉 Dataset ready at: {prepared_dataset}")
else:
    print("❌ Dataset preparation failed!")
```
