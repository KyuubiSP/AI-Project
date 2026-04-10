# CIFAR-10 Data Loading and Tensor Reshaping Explained

## Overview

This document explains how the CIFAR-10 dataset is loaded, preprocessed, and structured into PyTorch tensors. The code transforms raw binary batch files into properly formatted image tensors ready for neural network training.


---

## Understanding CIFAR-10 Data Format

### What is CIFAR-10?

CIFAR-10 (Canadian Institute For Advanced Research) is a dataset of 60,000 small images:

* **Training:** 50,000 images (5 batch files × 10,000 images each)
* **Testing:** 10,000 images (1 batch file)
* **Classes:** 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
* **Image Size:** 32×32 pixels
* **Color:** RGB (3 channels)

### Binary Format in Repository

Each batch file is stored as a pickled dictionary with this structure:

```python
{
    'batch_label': 'training batch 1 of 5',
    'labels': [6, 9, 9, 4, ...],           # class indices (0-9)
    'data': [[...3072 numbers...], ...],   # flattened images
    'filenames': ['leptodactylus_pentad...', ...]
}
```


---

## Code Breakdown: Loading and Reshaping

### Step 1: Load Binary File

```python
def load_batch(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')
```

**What it does:**

* Opens the binary pickle file
* Deserializes it back into a Python dictionary
* Uses `encoding='latin1'` because the file was pickled with Python 2

**Output:** Dictionary with `data`, `labels`, `batch_label`, and `filenames` keys


---

### Step 2: Convert to NumPy Array

```python
batch = load_batch(base / f'data_batch_1')
images = np.asarray(batch['data'], dtype=np.uint8)
```

**Image structure at this point:**

```
Shape: (10000, 3072)
Data Type: uint8 (unsigned integers 0-255)

10000 images × 3072 values per image
```

**Why 3072?**

* Each image: 32 × 32 × 3 (height × width × channels)
* Total: 32 × 32 × 3 = 3,072 values

**Important:** The data is stored as:

```
[R_pixels (1024), G_pixels (1024), B_pixels (1024)]
```

All red pixels first, then all green, then all blue (not interleaved per pixel).


---

### Step 3: Convert to PyTorch Tensor

```python
images = torch.from_numpy(images)
```

**Output:**

```
Shape: (10000, 3072)
Data Type: torch.uint8
Device: CPU
```

**Why convert from NumPy?**

* PyTorch models require PyTorch tensors
* Enables GPU computation
* Allows backpropagation with gradients


---

### Step 4: Reshape the Tensor (THE KEY STEP)

```python
images = images.view(-1, 3, 32, 32)
```

**Before reshape:**

```
Shape: (10000, 3072)
Each image: [R1, R2, ..., R1024, G1, G2, ..., G1024, B1, B2, ..., B1024]
```

**The** `-1` explained:

* `-1` tells PyTorch to automatically calculate this dimension
* Total elements must stay the same:
  * Before: 10000 × 3072 = 30,720,000 elements
  * After: ? × 3 × 32 × 32 = 30,720,000 elements
  * Therefore: ? = 30,720,000 ÷ (3 × 32 × 32) = 10,000

**After reshape:**

```
Shape: (10000, 3, 32, 32)

Dimension 0 (10000): Batch size = 10,000 images
Dimension 1 (3):     Color channels = [Red, Green, Blue]
Dimension 2 (32):    Image height in pixels
Dimension 3 (32):    Image width in pixels
```

**Visual representation of ONE image:**

```
Before:  [R, R, R, ..., R(1024 values), G, G, ..., G(1024 values), B, B, ..., B(1024 values)]

After:   [[[R00, R01, R02, ..., R31],
            [R10, R11, R12, ..., R31],
            ...
            [R31, R31, R31, ..., R31]],

           [[G00, G01, G02, ..., G31],
            [G10, G11, G12, ..., G31],
            ...
            [G31, G31, G31, ..., G31]],

           [[B00, B01, B02, ..., B31],
            [B10, B11, B12, ..., B31],
            ...
            [B31, B31, B31, ..., B31]]]

Shape of one image: (3, 32, 32)
```


---

### Step 5: Convert Data Type

```python
images = images.float()
```

**Before:**

* Data Type: `torch.uint8`
* Values: 0-255 (integers)

**After:**

* Data Type: `torch.float32`
* Values: 0-255.0 (floats)

**Why?**

* Neural networks need floating-point math for gradients
* Integer operations would lose precision during backpropagation


---

### Step 6: Normalize Pixel Values

```python
images = images / 255.0
```

**Before normalization:**

* Pixel values: 0 to 255 (large magnitude)
* Example bright pixel: 255.0

**After normalization:**

* Pixel values: 0 to 1 (small magnitude)
* Example bright pixel: 1.0

**Why normalize?**


1. **Numerical Stability:** Smaller values prevent exploding gradients
2. **Faster Training:** Neural networks converge faster with normalized inputs
3. **Standard Practice:** Matches ImageNet normalization conventions
4. **Better Accuracy:** Pre-trained models often expect this format

**Formula:**

```
normalized_pixel = original_pixel / 255.0
```


---

## Full Pipeline Summary

```
Binary File (.pkl)
       ↓
Unpickle → Python Dict with 'data' key
       ↓
Extract batch['data'] → NumPy array (10000, 3072) uint8
       ↓
torch.from_numpy() → PyTorch tensor (10000, 3072) uint8
       ↓
.view(-1, 3, 32, 32) → Reshape to proper image format (10000, 3, 32, 32) uint8
       ↓
.float() → Convert to float32 (10000, 3, 32, 32) float32
       ↓
/ 255.0 → Normalize to [0, 1] range (10000, 3, 32, 32) float32 [0-1]
       ↓
Ready for PyTorch Model ✓
```


---

## Complete Code Example

```python
import pickle
from pathlib import Path
import numpy as np
import torch

base = Path('cifar-10-batches-dataset')

def load_batch(path):
    """Load and deserialize a CIFAR-10 batch file"""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

# Example: Load and process one training batch
batch = load_batch(base / 'data_batch_1')

# Step-by-step transformation
images = np.asarray(batch['data'], dtype=np.uint8)           # (10000, 3072) uint8
print(f"After NumPy conversion: {images.shape}, {images.dtype}")

images = torch.from_numpy(images)                            # (10000, 3072) uint8
print(f"After torch.from_numpy(): {images.shape}, {images.dtype}")

images = images.view(-1, 3, 32, 32)                          # (10000, 3, 32, 32) uint8
print(f"After .view(-1, 3, 32, 32): {images.shape}, {images.dtype}")

images = images.float()                                       # (10000, 3, 32, 32) float32
print(f"After .float(): {images.shape}, {images.dtype}")

images = images / 255.0                                       # (10000, 3, 32, 32) float32 [0-1]
print(f"After normalization: {images.shape}, {images.dtype}, values in [{images.min()}, {images.max()}]")

# Access one image
one_image = images[0]  # Shape: (3, 32, 32)
print(f"Single image shape: {one_image.shape}")
```


---

## Using with CIFAR10Dataset Class

```python
from torch.utils.data import Dataset, DataLoader

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, label_names=None, transform=None):
        self.images = images           # Already normalized (B, 3, 32, 32)
        self.labels = labels           # Class indices (B,)
        self.label_names = label_names # Optional class names
        self.transform = transform     # Optional data augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]       # Get (3, 32, 32) tensor
        label = self.labels[idx]       # Get scalar label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label            # Return (image, label) tuple

# Create dataset
train_dataset = CIFAR10Dataset(train_images, train_labels, label_names)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Usage in training loop
for batch_images, batch_labels in train_loader:
    # batch_images shape: (32, 3, 32, 32)
    # batch_labels shape: (32,)
    print(batch_images.shape, batch_labels.shape)
    break  # Just show first batch
```


---

## Key Takeaways

| Concept | Explanation |
|----|----|
| **Batch Dimension** | First dimension (10000) is batch size, always present for vectorized operations |
| **Channel Dimension** | Second dimension (3) represents RGB color channels |
| **Spatial Dimensions** | Last two dimensions (32, 32) are height and width in pixels |
| **Data Type** | `uint8` for raw pixels, `float32` for normalized neural network inputs |
| **Normalization** | Dividing by 255.0 improves training stability and convergence |
| **View vs Reshape** | `.view()` reshapes without copying; only works if contiguous |
| **-1 Inference** | Automatically calculates dimension based on total element count |


---

## Common Issues and Solutions

### Issue: "RuntimeError: shape '\[...\] ' is invalid for input of size X"

**Cause:** Total elements don't match
**Solution:** Ensure `batch_size × 3 × 32 × 32 = original_total_elements`

### Issue: Tensor values out of range after operations

**Cause:** Forgot normalization step
**Solution:** Always divide by 255.0 after converting to float

### Issue: Model training doesn't converge

**Cause:** Unnormalized input (values 0-255 too large)
**Solution:** Verify `/255.0` step is applied


---

## References

* CIFAR-10 Dataset: https://www.cs.toronto.edu/\~kriz/cifar.html
* PyTorch Tensor Reshaping: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
* Data Normalization Best Practices: https://en.wikipedia.org/wiki/Feature_scaling


