# HFCF Branch Implementation for CFRWD-GAN

## Overview

This document describes the implementation of the High-Frequency Coding and Filtering (HFCF) branch in the CFRWD-GAN model, following the architecture described in the research paper:

**Citation**: Wei, J.; Zou, H.; Sun, L.; Cao, X.; He, S.; Liu, S.; Zhang, Y. "CFRWD-GAN for SAR-to-Optical Image Translation." Remote Sens. 2023, 15, 2547.

## Architecture Components

### 1. Discrete Wavelet Transform (DWT)

The HFCF branch begins with a 2-level Haar wavelet decomposition that separates the SAR image into different frequency components:

```
Input SAR Image: B × 1 × 256 × 256

After 2-level DWT:
├── g1 (LL₂): B × 1 × 64 × 64    [Low-frequency approximation]
├── g2 (LH₂, HL₂, HH₂): B × 3 × 64 × 64    [High-frequency details, level 2]
└── g3 (LH₁, HL₁, HH₁): B × 3 × 128 × 128  [High-frequency details, level 1]
```

**Purpose**: Separate noise from signal by decomposing into frequency bands. Speckle noise in SAR images primarily affects high-frequency components, which can be filtered while preserving structural details.

### 2. High-Frequency Component Preprocessing

Each high-frequency group (g2 and g3) undergoes preprocessing:

```python
HFCFPreprocess:
    Conv2d(3→32, kernel=3, stride=1) + InstanceNorm + ReLU + MaxPool(2×2)
```

**Dimension changes**:
- g2: 64×64 → 32×32 (after MaxPool)
- g3: 128×128 → 64×64 (after MaxPool)

### 3. Upper Branch (ResNet101-style)

Processes g2 (higher compression, more abstract features) through Bottleneck blocks:

```
Architecture: Yellow → Blue → Blue → Yellow → Blue → Blue

Input:  32 channels × 32×32
├── Yellow Block 1: 32→64 channels, spatial ÷2 → 64 × 16×16
├── Blue Block 1:   64→64 channels, maintain  → 64 × 16×16
├── Blue Block 2:   64→64 channels, maintain  → 64 × 16×16
├── Yellow Block 2: 64→128 channels, spatial ÷2 → 128 × 8×8
├── Blue Block 3:   128→128 channels, maintain → 128 × 8×8
└── Blue Block 4:   128→128 channels, maintain → 128 × 8×8

Output: 128 channels × 8×8
```

**Block Structures**:
- **Yellow Block (Bottleneck with downsample)**: 3 convolutions with stride=2 in final layer, expands channels
- **Blue Block (Bottleneck)**: 3 convolutions maintaining resolution and channels

### 4. Lower Branch (ResNet18-style)

Processes g3 (higher resolution, more detailed features) through BasicBlock blocks:

```
Architecture: Red → Red

Input:  32 channels × 64×64
├── Red Block 1: 32→32 channels, maintain → 32 × 64×64
└── Red Block 2: 32→32 channels, maintain → 32 × 64×64

Output: 32 channels × 64×64
```

**Block Structure**:
- **Red Block (BasicBlock)**: 2 convolutions maintaining resolution and channels

### 5. Spatial Alignment

Before concatenation, outputs must have matching spatial dimensions:

```
Upper branch output: 128 channels × 8×8
Lower branch output: 32 channels × 64×64

Alignment (3× AvgPool2d with 2×2 kernel):
    64×64 → 32×32 → 16×16 → 8×8

Aligned lower output: 32 channels × 8×8
```

### 6. Concatenation

```
Concatenated: [128 + 32] channels × 8×8 = 160 channels × 8×8
```

### 7. Upconvolution and Reconstruction

Progressive upsampling to reconstruct the optical image:

```
Input: 160 channels × 8×8

├── Conv: 160→128 channels, maintain → 128 × 8×8
├── TConv1: 128→128 channels, upsample ×2 → 128 × 16×16
├── TConv2: 128→64 channels, upsample ×2 → 64 × 32×32
├── TConv3: 64→32 channels, upsample ×2 → 32 × 64×64
├── TConv4: 32→16 channels, upsample ×2 → 16 × 128×128
├── TConv5: 16→8 channels, upsample ×2 → 8 × 256×256
└── Final Conv: 8→3 channels + Tanh → 3 × 256×256

Output: B × 3 × 256 × 256 (RGB optical image)
```

## Design Rationale

### Why Two Branches?

1. **Upper Branch (g2)**: 
   - Processes lower-resolution, more compressed high-frequency features
   - Uses deeper, more abstract feature extraction (ResNet101-style)
   - Focuses on semantic structure and patterns

2. **Lower Branch (g3)**:
   - Processes higher-resolution high-frequency features
   - Uses simpler feature extraction (ResNet18-style)
   - Preserves fine-grained details and textures

### Why Different Block Types?

The paper uses ResNet101 (Bottleneck) and ResNet18 (BasicBlock) architectures because:

- **Bottleneck blocks** (Yellow/Blue): More parameters, better for extracting complex semantic features from compressed representations
- **BasicBlock blocks** (Red): Fewer parameters, better for preserving spatial details without over-processing

### Alignment Strategy

The aggressive downsampling of the lower branch (64×64 → 8×8) might seem counterintuitive, but:

1. The lower branch has already extracted relevant high-frequency patterns
2. The upconvolution stage reconstructs fine details
3. This design reduces computational cost while maintaining quality

## Key Implementation Details

### 1. Block Implementations

**YellowBlock** (Bottleneck with stride=2):
```python
3 Conv layers with InstanceNorm + LeakyReLU
- Conv1: 3×3, stride=1
- Conv2: 3×3, stride=1  
- Conv3: 3×3, stride=2 (downsamples)
+ Skip connection with 1×1 Conv, stride=2
```

**BlueBlock** (Bottleneck maintaining resolution):
```python
3 Conv layers with InstanceNorm + LeakyReLU
- All 3×3, stride=1
+ Skip connection (identity)
```

**RedBlock** (BasicBlock):
```python
2 Conv layers with InstanceNorm + LeakyReLU
- Both 3×3, stride=1
+ Skip connection (identity)
```

### 2. Activation Functions

- **Inside blocks**: LeakyReLU(0.2) - allows small negative gradients, helps training stability
- **No activation at block output**: Allows residual connections to work properly
- **Final layer**: Tanh - maps output to [-1, 1] range for image generation

### 3. Normalization

- **InstanceNorm2d**: Used throughout (better for style transfer tasks)
- Affine=True: Allows learning scale and shift parameters

### 4. Padding

- **ReflectionPad2d**: Used instead of zero-padding to avoid border artifacts

## Testing the Implementation

To verify the implementation works correctly, you can trace through the dimensions:

```python
import torch
from src.models.cfrwd.gen import CFRWDGenerator

# Create model
model = CFRWDGenerator(in_channels=1)

# Test input
x = torch.randn(1, 1, 256, 256)

# Forward pass
output = model(x)

# Check output shape
assert output.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {output.shape}"
print("✓ HFCF branch implementation verified!")
```

## Comparison with Paper

Our implementation follows the CFRWD paper's specifications:

✓ 2-level Haar wavelet decomposition  
✓ Separate processing of g2 and g3  
✓ ResNet101-style blocks for upper branch  
✓ ResNet18-style blocks for lower branch  
✓ Proper spatial alignment before concatenation  
✓ Progressive upsampling to original resolution  

## Future Improvements

Potential enhancements to consider:

1. **Adaptive alignment**: Instead of aggressive downsampling, use learned downsampling or attention mechanisms
2. **Multi-scale fusion**: Incorporate features from multiple scales in the upsampling stage
3. **Attention mechanisms**: Add self-attention in the branches to focus on important regions
4. **Skip connections**: Add skip connections from DWT components to upsampling layers

## References

1. Wei, J. et al. (2023). "CFRWD-GAN for SAR-to-Optical Image Translation." Remote Sensing, 15, 2547.
2. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
3. Wang, J. et al. (2020). "Deep High-Resolution Representation Learning for Visual Recognition." TPAMI.
