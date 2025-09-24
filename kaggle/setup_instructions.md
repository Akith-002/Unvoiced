# Step-by-Step Kaggle Setup Instructions

## 1. Prepare Your Dataset

### Upload Dataset to Kaggle Datasets

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload your `asl_alphabet_train` folder (the one with A/, B/, C/, etc. subdirectories)
4. Set dataset title: "ASL Alphabet Training Dataset"
5. Set visibility to "Public" for easy access
6. Add description and tags
7. Click "Create"
8. **Note the dataset URL** - you'll need this!

### Alternative: Use Existing Dataset

- Search for "ASL alphabet" on Kaggle datasets
- Look for datasets with ~3000 images per letter
- Add it to your notebook

## 2. Create Kaggle Notebook

### Setup Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Choose "Notebook" (not Script)
4. **IMPORTANT**: Set accelerator to "GPU T4 x2" or "GPU P100"
5. Set language to "Python"

### Add Dataset

1. In your notebook, click "Add Data" on the right
2. Search for your uploaded dataset
3. Click "Add" to attach it to your notebook
4. Note the path (usually `/kaggle/input/your-dataset-name/`)

## 3. Setup and Run Training

### Method A: Direct Code Execution (Recommended)

1. Copy the code from `notebook_cells.md`
2. Create separate cells for each section
3. Run cells in order
4. Training will take 2-3 hours with GPU

### Method B: Upload Script File

1. In your notebook, click "Add Data"
2. Upload the `kaggle_train.py` file
3. Run it with: `exec(open('/kaggle/input/your-script/kaggle_train.py').read())`

## 4. Monitor Training Progress

### Check Progress

- Training will show live progress with epoch information
- Look for lines like: `Epoch 1/30 - loss: 2.1234 - accuracy: 0.6789`
- GPU training should complete 1 epoch every 4-6 minutes

### Expected Timeline

- **Dataset preparation**: 5-10 minutes
- **Model creation**: 2-3 minutes
- **Phase 1 training** (15 epochs): 60-90 minutes
- **Phase 2 fine-tuning** (15 epochs): 60-90 minutes
- **Evaluation & export**: 10-15 minutes
- **Total**: 2.5-3.5 hours

## 5. Download Results

### What You'll Get

After training completes, you'll have in `/kaggle/working/models/`:

- `model.keras` - Full Keras model (50-100MB)
- `model.tflite` - Mobile-optimized model (15-25MB)
- `frozen_model.pb` - Frozen graph for your original code (50-100MB)
- `training_set_labels.txt` - Class labels file
- `evaluation_report.json` - Detailed performance metrics
- `training_history.json` - Training progress data
- `confusion_matrix.png` - Visual performance analysis

### Download Files

1. After training completes, files will be in `/kaggle/working/models/`
2. Use the zip creation cell to package everything
3. Download the zip file from Kaggle
4. Extract and use in your Flutter app!

## 6. Integration with Your Flask Backend

### Replace Original Model Files

1. Download `frozen_model.pb` and `training_set_labels.txt`
2. Replace the files in your original project:
   - `D:\Projects\Unvoiced\trained_model_graph.pb` → `frozen_model.pb`
   - `D:\Projects\Unvoiced\training_set_labels.txt` → `training_set_labels.txt`
3. Your Flask backend should work with the new model!

### Use TFLite for Mobile

- Use `model.tflite` in your Flutter app for better mobile performance
- It's smaller and optimized for mobile inference

## 7. Troubleshooting

### Common Issues

- **No GPU**: Enable GPU accelerator in notebook settings
- **Dataset not found**: Check dataset path in `/kaggle/input/`
- **Out of memory**: Reduce batch size in the script
- **Slow training**: Ensure GPU is enabled and being used

### Performance Expectations

- **With GPU**: 2-3 hours for 30 epochs
- **Without GPU**: 12-16 hours (not recommended)
- **Expected accuracy**: 85-95% on test set

### Getting Help

- Check Kaggle's discussion forums
- Look at the notebook's logs for error messages
- Monitor GPU usage in Kaggle's resource panel

## 8. Optional Optimizations

### For Better Results

- Increase epochs to 50 for production use
- Experiment with different image sizes
- Try different augmentation techniques
- Use larger batch sizes if memory allows

### For Faster Training

- Use smaller image size (150x150 instead of 200x200)
- Reduce number of classes if not all letters are needed
- Use MobileNetV2 instead of V3 for faster training
