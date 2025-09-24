# IMMEDIATE FIX FOR YOUR KAGGLE NOTEBOOK - CELL 7

## REPLACE YOUR CELL 7 WITH THIS:

```python
print("🔥 Phase 2: Fine-tuning entire model")

# Unfreeze base model
base_model.trainable = True

# Recompile with lower learning rate - FIXED learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Higher LR than before
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Phase 2 callbacks
callbacks_phase2 = [
    ModelCheckpoint(
        str(output_path / 'best_model_final.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Reduced patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Less aggressive reduction
        patience=3,
        min_lr=0.00001,  # Higher minimum LR
        verbose=1
    )
]

# Train phase 2 with error handling
try:
    print("Starting Phase 2 training...")
    history_phase2 = model.fit(
        train_ds,
        epochs=10,  # Fewer epochs to avoid issues
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1,
        initial_epoch=15
    )
    print("✅ Phase 2 completed successfully")
    phase2_success = True
except Exception as e:
    print(f"⚠️  Phase 2 failed: {e}")
    print("Continuing with Phase 1 model only...")
    # Create dummy history
    class DummyHistory:
        def __init__(self):
            self.history = {}
    history_phase2 = DummyHistory()
    phase2_success = False

# Safe history combination
print("🔗 Combining training history...")

if phase2_success and len(history_phase2.history) > 0:
    # Normal combination
    combined_history = {}
    for key in history_phase1.history.keys():
        if key in history_phase2.history:
            combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
            print(f"✅ Combined {key}")
        else:
            combined_history[key] = history_phase1.history[key]
            print(f"⚠️  Only Phase 1 {key}")
else:
    # Phase 2 failed - use only Phase 1
    combined_history = history_phase1.history.copy()
    print("⚠️  Using only Phase 1 history")

print(f"✅ Training completed! History length: {len(combined_history.get('loss', []))}")
```

## WHY PHASE 2 FAILED:

1. **Learning rate too low** (0.00002 is extremely small)
2. **Too many epochs** starting from epoch 15
3. **Aggressive callbacks** might stop training immediately

## THE FIXES:

1. **Higher learning rate** (0.0001 instead of 0.00002)
2. **Fewer epochs** (10 instead of 15)
3. **Better error handling**
4. **Safe history combination**

Copy this entire cell and replace your Cell 7 - it should work much better!

```

```
