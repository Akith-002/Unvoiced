#!/usr/bin/env python3
"""
GPU Detection Script
Check if TensorFlow can detect and use GPU
"""

try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
    # List physical devices
    print(f"\nPhysical devices:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device}")
    
    # Check specifically for GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices: {len(gpu_devices)}")
    for i, gpu in enumerate(gpu_devices):
        print(f"  GPU {i}: {gpu}")
    
    # Test GPU availability
    if len(gpu_devices) > 0:
        print("\n✅ GPU is available!")
        # Try a simple computation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = tf.add(a, b)
            print(f"GPU computation test: {c.numpy()}")
    else:
        print("\n❌ No GPU detected - will use CPU")
        
    # Check available devices for computation
    print(f"\nAvailable devices for computation:")
    for device in tf.config.list_logical_devices():
        print(f"  {device}")
    
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
except Exception as e:
    print(f"Error checking GPU: {e}")