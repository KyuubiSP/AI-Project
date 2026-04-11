# Car vs Truck Modeling Explanation

## Overview

This document explains the modeling notebook for the car-vs-truck classifier. The goal is to train three PyTorch image classification models on a filtered CIFAR-style dataset containing only two classes:

* `car`
* `truck`

The notebook assumes `train_dataset` and `test_dataset` are already prepared by teammates, so the modeling part starts directly from ready-to-use PyTorch datasets.


## Input Assumption

The notebook expects:

```python
train_dataset
test_dataset
```

Both should behave like standard PyTorch datasets:

```python
image, label = train_dataset[0]
```

Expected format:

* `image`: tensor with shape `(3, H, W)`
* `label`: integer class index
* labels should represent the two classes only

The notebook is written so it can handle images already normalized to `[0, 1]` or still stored in `[0, 255]`. If values are above `1`, it divides them by `255.0`.


## Why Three Models Are Used

The notebook trains three different models for the same task:

1. a small custom CNN
2. a pretrained `ResNet18`
3. a pretrained `Swin-Tiny`

This gives a simple baseline plus two stronger transfer-learning models built from different pretrained backbones.

The custom CNN is useful because:

* it is easy to explain
* it is small and fast
* it shows how a CNN can be built from scratch

The `ResNet18` model is useful because:

* it starts from pretrained image features
* it usually performs better than a small CNN on limited data
* it is a standard transfer-learning approach

The `Swin-Tiny` model is useful because:

* it is a transformer-based vision model instead of a CNN
* it brings a second transfer-learning option with a different architecture
* it can capture image structure with shifted-window self-attention


## Model 1: Custom CNN Baseline

The baseline model is a straightforward convolutional neural network with three convolution blocks.

Architecture idea:

```text
Input image
→ Conv2d
→ ReLU
→ MaxPool
→ Conv2d
→ ReLU
→ MaxPool
→ Conv2d
→ ReLU
→ MaxPool
→ AdaptiveAvgPool
→ Flatten
→ Linear
→ ReLU
→ Dropout
→ Linear(2 outputs)
```

### Why this design?

* Convolution layers learn local visual patterns such as edges and shapes.
* ReLU adds non-linearity.
* Max pooling reduces spatial size and keeps important features.
* The final linear layer outputs 2 logits, one for each class.

`CrossEntropyLoss` expects raw logits, so adding `softmax` inside the model would be the wrong setup for training.


## Model 2: ResNet18 Transfer Learning

The second model uses:

```python
torchvision.models.resnet18
```

with pretrained ImageNet weights.

The final classification layer is replaced:

```python
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
```

This changes the network from ImageNet classification to the car-vs-truck task.

### Fine-tuning strategy

The notebook fine-tunes the whole network:

* no layers are frozen
* all parameters are trainable
* the optimizer updates the full model

This means the pretrained features are adapted to the new dataset instead of using the network only as a fixed feature extractor.


## Model 3: Swin-Tiny Transfer Learning

The third model uses:

```python
torchvision.models.swin_t
```

with pretrained ImageNet weights.

The classification head is replaced:

```python
swin_model.head = nn.Linear(swin_model.head.in_features, 2)
```

This changes the original ImageNet output layer to a binary car-vs-truck classifier.

### Fine-tuning strategy

The notebook fine-tunes the whole `Swin-Tiny` model:

* no layers are frozen
* all parameters remain trainable
* Adam updates the full transformer backbone and the new head

This makes the notebook's Swin-Tiny setup parallel to the ResNet18 setup.


## Why ResNet18 and Swin-Tiny Need Extra Preprocessing

The baseline CNN can work directly on small CIFAR-like images, but both `ResNet18` and `Swin-Tiny` were originally trained on ImageNet images.

For that reason, the notebook prepares the input for both pretrained models by:

1. resizing images to `224 x 224`
2. normalizing them with ImageNet mean and standard deviation

Code idea:

```python
images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
images = (images - mean) / std
```

This makes the dataset more compatible with the pretrained networks.


## Loss Function

All models use:

```python
nn.CrossEntropyLoss()
```

This is appropriate because:

* the task is classification
* there are two classes
* the model outputs 2 logits

Even though this is a binary problem, using 2 logits with cross-entropy is a standard and correct approach.


## Optimizer

The notebook uses the Adam optimizer:

```python
torch.optim.Adam(...)
```

Why Adam:

* simple to use
* works well as a default optimizer
* needs little manual tuning for a class project

Different learning rates are used for the three models:

* custom CNN: higher learning rate
* ResNet18: lower learning rate
* Swin-Tiny: lower learning rate

This is common because pretrained models are usually fine-tuned more carefully than a small CNN trained from scratch.


## Training Loop

The notebook follows a standard PyTorch training loop:

1. load a batch from the dataloader
2. move images and labels to the selected device
3. run the forward pass
4. compute loss
5. clear old gradients
6. run backpropagation
7. update parameters

Simplified structure:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

During training, the notebook also computes:

* training loss
* training accuracy
* test loss
* test accuracy

These values are stored after each epoch.


## Evaluation

Evaluation is done with:

```python
model.eval()
with torch.no_grad():
```

Why:

* `model.eval()` disables training-only behavior such as dropout
* `torch.no_grad()` saves memory and speeds up evaluation

Accuracy is computed by taking the class with the highest logit:

```python
predictions = outputs.argmax(dim=1)
```


## Saved Outputs

After training, all models are saved as PyTorch state dicts:

* `saved_models/simple_cnn_car_truck.pth`
* `saved_models/resnet18_car_truck.pth`
* `saved_models/swin_tiny_car_truck.pth`

The notebook also includes example loading code so teammates can reload the weights later for testing or evaluation.

This is done with:

```python
torch.save(model.state_dict(), path)
```

and later:

```python
model.load_state_dict(torch.load(path))
```

Saving the state dict is better than saving the whole model object because it is the standard PyTorch approach and is more portable.


## Summary

The modeling notebook does the following:

1. receives already-prepared `train_dataset` and `test_dataset`
2. creates dataloaders
3. trains a simple CNN baseline
4. trains a fine-tuned pretrained `ResNet18`
5. trains a fine-tuned pretrained `Swin-Tiny`
6. logs loss and accuracy for all models
7. plots training progress
8. saves all trained models for later evaluation

In short, the notebook gives one from-scratch baseline plus two transfer-learning models for binary image classification on the car-vs-truck subset.
