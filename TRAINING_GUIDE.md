# Training Guide for ASL Sign Language Model

## Training Pipeline Overview

This training pipeline uses **MobileNetV3Large** optimized for mobile deployment with your 6GB GPU. The pipeline includes:

- ✅ **GPU memory optimization** for 6GB cards
- ✅ **Mixed precision training** for faster training
- ✅ **Two-phase fine-tuning** (frozen → unfrozen)
- ✅ **Data augmentation** for better generalization
- ✅ **Class balancing** with automatic weight calculation
- ✅ **Multiple export formats**: SavedModel, TFLite, Frozen Graph

## Quick Start

### 1. Install Training Dependencies

```powershell
pip install -r backend/requirements_training.txt
```

### 2. Prepare Your Dataset

If you have a single folder with 87k images organized by class:

```powershell
python backend/prepare_dataset.py --input_dir path/to/your/87k_images --output_dir dataset
```

This creates:

```
dataset/
├── train/     (70% = ~61k images)
├── val/       (15% = ~13k images)
└── test/      (15% = ~13k images)
```

### 3. Start Training

```powershell
python backend/train_finetune.py --dataset_dir dataset --epochs 50 --batch_size 32
```

**Expected runtime on 6GB GPU**: ~3-4 hours for 50 epochs

### 4. Monitor Training

- Watch console for epoch progress
- Training curves saved to `output/training_history.png`
- TensorBoard logs: `tensorboard --logdir output/logs` (if implemented)

## GPU Memory Settings for 6GB

The script automatically:

- Enables GPU memory growth (avoids OOM)
- Uses mixed precision (faster training)
- Optimizes batch size for 6GB cards
- If you get OOM, reduce `--batch_size` to 16 or 24

## Training Results

After training completes, you'll find in `output/`:

### For Backend Server

- `saved_model/` - Load with `tf.saved_model.load()`
- `trained_model_graph.pb` - Frozen graph (compatible with existing backend)
- `training_set_labels.txt` - Class names in prediction order

### For Mobile App

- `model.tflite` - Optimized for mobile (typically 10-15MB)
- Size optimized for Flutter/mobile deployment

### Evaluation

- `evaluation_report.json` - Per-class accuracy, precision, recall
- `confusion_matrix.png` - Visual confusion matrix
- `training_history.png` - Loss/accuracy curves

## Typical Results with 87k Images

Expected performance:

- **Test Accuracy**: 95-98% (with good quality data)
- **Per-class accuracy**: 90%+ for most letters
- **Problematic pairs**: I/J, M/N, similar gestures
- **TFLite size**: ~12-15MB (mobile-ready)

## Advanced Usage

### Custom Hyperparameters

```powershell
# Smaller batch for limited GPU memory
python backend/train_finetune.py --batch_size 16

# More epochs for better accuracy
python backend/train_finetune.py --epochs 100

# Different image size
python backend/train_finetune.py --img_size 224 224
```

### Using the Trained Model

#### Replace Backend Model

```powershell
# Copy new frozen graph to replace existing
copy output/trained_model_graph.pb trained_model_graph.pb
copy output/training_set_labels.txt training_set_labels.txt
```

#### Test New Model

```powershell
# Test with new model (set USE_MOCK_MODEL=0)
python backend/run_smoke_test.py
```

## Troubleshooting

### GPU Issues

- **OOM Error**: Reduce `--batch_size` to 16 or 24
- **CUDA Error**: Update GPU drivers, check TensorFlow-GPU compatibility
- **No GPU Detected**: Install CUDA toolkit + cuDNN

### Dataset Issues

- **No images found**: Check file extensions (.jpg, .png, .jpeg)
- **Class imbalance**: Script automatically calculates class weights
- **Low accuracy**: Verify image quality, increase epochs, check augmentation

### Memory Optimization for 6GB GPU

```python
# The script automatically applies these optimizations:
tf.config.experimental.set_memory_growth(gpu, True)  # Dynamic memory
policy = tf.keras.mixed_precision.Policy('mixed_float16')  # 2x faster
```

## Cost Comparison: Local vs Cloud

### Your 6GB GPU (Free)

- ✅ **Cost**: $0
- ✅ **Speed**: 3-4 hours for 87k images
- ✅ **Privacy**: Data stays local
- ❌ **Limitation**: Single GPU

### Cloud GPU ($0.50-2.00/hour)

- ❌ **Cost**: ~$2-8 for full training
- ✅ **Speed**: Potentially faster with V100/A100
- ❌ **Privacy**: Data uploaded
- ✅ **Scalability**: Multiple GPUs

**Recommendation**: Use your local 6GB GPU — it's perfectly capable and free!

## Production Deployment

### Backend Integration

1. Copy `output/saved_model/` to replace existing model
2. Update `backend/app.py` to load SavedModel instead of frozen graph
3. Copy `output/training_set_labels.txt` to repo root

### Mobile Integration

1. Copy `output/model.tflite` to Flutter assets
2. Use `tflite_flutter` package in Flutter
3. Implement same preprocessing (200x200, normalize to [0,1])

### Real-time Improvements

- Add temporal smoothing (majority vote over 3-5 frames)
- Use confidence thresholds (only accept predictions > 0.7)
- Implement gesture sequence validation

Ready to start training? Run the preparation command first, then training!
