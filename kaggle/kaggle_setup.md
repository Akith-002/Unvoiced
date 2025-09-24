# Kaggle Training Setup Guide

## Overview

Train your ASL model on Kaggle with free GPU access (30h/week) instead of slow CPU training.

## Prerequisites

1. Kaggle account (free)
2. Upload your dataset to Kaggle Datasets
3. Create a new Kaggle Notebook

## Expected Performance

- **CPU (Local)**: 8-12 hours for 20 epochs
- **GPU (Kaggle)**: 2-3 hours for 20 epochs
- **GPU Available**: Tesla P100, T4, or newer

## Setup Steps

### 1. Upload Dataset to Kaggle

- Go to kaggle.com/datasets
- Click "New Dataset"
- Upload your `asl_alphabet_train` folder
- Make it public for easy access

### 2. Create Kaggle Notebook

- Go to kaggle.com/code
- Click "New Notebook"
- Enable GPU accelerator
- Add your dataset as data source

### 3. Run the provided code cells

### 4. Download trained models

- Models will be saved in `/kaggle/working/`
- Download the results when training completes

## Files to Create in Kaggle

1. `kaggle_train.py` - Main training script
2. `requirements.txt` - Dependencies
3. Data preparation cells
4. Training execution cells
5. Model export cells

## Directory Structure in Kaggle

```
/kaggle/input/asl-dataset/asl_alphabet_train/
    A/
    B/
    ...
/kaggle/working/
    models/
        model.keras
        model.tflite
        frozen_model.pb
        training_set_labels.txt
```
