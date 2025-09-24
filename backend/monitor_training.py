"""
Simple script to monitor training progress by checking output directory
"""
import os
import time
import json
from pathlib import Path

def monitor_training(output_dir="d:/Projects/Unvoiced/output", check_interval=30):
    output_path = Path(output_dir)
    
    print(f"🔍 Monitoring training progress in: {output_path}")
    print(f"📊 Checking every {check_interval} seconds...")
    print("-" * 50)
    
    while True:
        try:
            # Check if output directory exists
            if not output_path.exists():
                print("⏳ Waiting for training to start...")
                time.sleep(check_interval)
                continue
            
            # Check for model files
            models = list(output_path.glob("*.keras"))
            if models:
                print(f"📈 Found model checkpoints: {len(models)}")
                for model in models:
                    size_mb = model.stat().st_size / (1024 * 1024)
                    mod_time = time.ctime(model.stat().st_mtime)
                    print(f"   {model.name}: {size_mb:.1f}MB (modified: {mod_time})")
            
            # Check for training history
            history_file = output_path / "training_history.json"
            if history_file.exists():
                try:
                    with open(history_file) as f:
                        history = json.load(f)
                    
                    if 'loss' in history and history['loss']:
                        epochs_completed = len(history['loss'])
                        last_loss = history['loss'][-1]
                        last_acc = history.get('accuracy', [0])[-1] if history.get('accuracy') else 0
                        last_val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
                        last_val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
                        
                        print(f"🏃 Epochs completed: {epochs_completed}")
                        print(f"📉 Loss: {last_loss:.4f} | Accuracy: {last_acc:.4f}")
                        print(f"📊 Val Loss: {last_val_loss:.4f} | Val Accuracy: {last_val_acc:.4f}")
                        
                except Exception as e:
                    print(f"⚠️  Could not read training history: {e}")
            
            # Check for final results
            if (output_path / "evaluation_report.json").exists():
                print("🎉 Training completed! Evaluation report found.")
                try:
                    with open(output_path / "evaluation_report.json") as f:
                        results = json.load(f)
                    print(f"🎯 Final Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
                except Exception as e:
                    print(f"⚠️  Could not read evaluation results: {e}")
                break
            
            print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')} - Training in progress...")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    monitor_training()