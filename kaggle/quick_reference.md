# Kaggle Training Quick Reference

## 🚀 Quick Start (30 seconds)

1. Upload dataset to Kaggle Datasets
2. Create new notebook with GPU enabled
3. Add your dataset as data source
4. Copy code from `notebook_cells.md`
5. Run cells in order
6. Wait 2-3 hours
7. Download results!

## 📁 Required Files

- `kaggle_train.py` - Complete training script
- `notebook_cells.md` - Jupyter notebook cells
- Your `asl_alphabet_train/` dataset

## ⚡ Performance Comparison

| Environment  | Hardware                    | Time       |
| ------------ | --------------------------- | ---------- |
| **Local PC** | AMD RX 6600M (CPU fallback) | 8-12 hours |
| **Kaggle**   | Tesla P100/T4 GPU           | 2-3 hours  |

## 🎯 Expected Results

- **Test Accuracy**: 85-95%
- **Model Size**: 50-100MB (Keras), 15-25MB (TFLite)
- **Classes**: 29 (A-Z + space + delete + nothing)

## 📦 Output Files

```
/kaggle/working/models/
├── model.keras              # Full model
├── model.tflite            # Mobile optimized
├── frozen_model.pb         # For your Flask backend
├── training_set_labels.txt # Class names
├── evaluation_report.json  # Performance metrics
├── training_history.json   # Training progress
└── confusion_matrix.png    # Visual analysis
```

## 🔧 Kaggle Settings

- **Accelerator**: GPU T4 x2 (or P100)
- **Language**: Python
- **Internet**: On (for downloading packages)
- **Dataset**: Your uploaded ASL dataset

## 🐛 Quick Fixes

- **No GPU detected**: Enable GPU in notebook settings
- **Dataset not found**: Check path in `/kaggle/input/your-dataset-name/`
- **Out of memory**: Reduce batch_size in script
- **Slow training**: Verify GPU is enabled and active

## 📱 Flutter Integration

Replace these files in your project:

- `trained_model_graph.pb` ← `frozen_model.pb`
- `training_set_labels.txt` ← `training_set_labels.txt`

## 💡 Pro Tips

- Enable internet for package downloads
- Use GPU T4 x2 for fastest training
- Download files before notebook auto-saves
- Save intermediate checkpoints
- Monitor GPU usage in resource panel
