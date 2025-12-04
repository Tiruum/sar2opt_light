# HFCF Branch Data Flow Diagram

## Complete Data Flow

```
SAR Input (B×1×256×256)
         |
         v
    [DWT Block]
         |
    ┌────┴────┬─────────┐
    |         |         |
    v         v         v
   g1        g2        g3
  (LL₂)   (LH₂,HL₂,HH₂) (LH₁,HL₁,HH₁)
 1×64×64   3×64×64    3×128×128
    |         |         |
    |         v         v
    |   [Preprocess] [Preprocess]
    |     Conv+Pool   Conv+Pool
    |         |         |
    |    32×32×32   32×64×64
    |         |         |
    |         v         v
    |   [UpperBranch] [LowerBranch]
    |    ResNet101    ResNet18
    |    Y→B→B→Y→B→B  R→R
    |         |         |
    |    128×8×8    32×64×64
    |         |         |
    |         |         v
    |         |    [AvgPool×3]
    |         |         |
    |         |     32×8×8
    |         |         |
    |         └────┬────┘
    |              v
    |        [Concatenate]
    |        160×8×8
    |              |
    |              v
    |      [Upconvolution]
    |       5× Upsample
    |              |
    |              v
    |         3×256×256
    |              |
    └──────────────┘ (Not used in current HFCF,
                      but available for future enhancement)
```

## Detailed Upper Branch Flow

```
Input: 32 channels × 32×32
         |
         v
    [Yellow Block 1]
    ┌────────────────┐
    │ Conv 3×3, s=1  │
    │ Conv 3×3, s=1  │
    │ Conv 3×3, s=2  │──┐
    └────────────────┘  │
    ┌────────────────┐  │
    │ Skip: Conv1×1  │──┤
    │      s=2       │  │
    └────────────────┘  │
         └───────┬──────┘
                 v
         64 ch × 16×16
                 |
                 v
    [Blue Block 1]
    ┌────────────────┐
    │ Conv 3×3, s=1  │
    │ Conv 3×3, s=1  │
    │ Conv 3×3, s=1  │──┐
    └────────────────┘  │
    ┌────────────────┐  │
    │ Skip: Identity │──┤
    └────────────────┘  │
         └───────┬──────┘
                 v
         64 ch × 16×16
                 |
                 v
    [Blue Block 2]
         (same structure)
                 |
                 v
         64 ch × 16×16
                 |
                 v
    [Yellow Block 2]
         (s=2 downsample)
                 |
                 v
        128 ch × 8×8
                 |
                 v
    [Blue Block 3]
                 |
                 v
        128 ch × 8×8
                 |
                 v
    [Blue Block 4]
                 |
                 v
        128 ch × 8×8
```

## Detailed Lower Branch Flow

```
Input: 32 channels × 64×64
         |
         v
    [Red Block 1]
    ┌────────────────┐
    │ Conv 3×3, s=1  │
    │ Conv 3×3, s=1  │──┐
    └────────────────┘  │
    ┌────────────────┐  │
    │ Skip: Identity │──┤
    └────────────────┘  │
         └───────┬──────┘
                 v
         32 ch × 64×64
                 |
                 v
    [Red Block 2]
         (same structure)
                 |
                 v
         32 ch × 64×64
```

## Upconvolution Detail

```
Input: 160 ch × 8×8
         |
         v
    [Conv 3×3, s=1]
    160→128 channels
         |
         v
    128 ch × 8×8
         |
         v
    [TConv 4×4, s=2]
    128→128 channels
         |
         v
    128 ch × 16×16
         |
         v
    [TConv 4×4, s=2]
    128→64 channels
         |
         v
     64 ch × 32×32
         |
         v
    [TConv 4×4, s=2]
    64→32 channels
         |
         v
     32 ch × 64×64
         |
         v
    [TConv 4×4, s=2]
    32→16 channels
         |
         v
     16 ch × 128×128
         |
         v
    [TConv 4×4, s=2]
    16→8 channels
         |
         v
      8 ch × 256×256
         |
         v
    [Conv 3×3 + Tanh]
    8→3 channels
         |
         v
      3 × 256×256
    (RGB Optical Image)
```

## Spatial Resolution Tracking

```
Stage                    Upper Branch    Lower Branch
─────────────────────────────────────────────────────
Input (SAR)              256×256         256×256
DWT Level 2              64×64           -
DWT Level 1              -               128×128
Preprocessing            32×32           64×64
Branch Processing:
  - After 1st block      16×16           64×64
  - After 2nd block      16×16           64×64
  - After 3rd block      16×16           -
  - After 4th block      8×8             -
  - After 5th block      8×8             -
  - After 6th block      8×8             -
Alignment                8×8             8×8 (pooled)
Concatenation            8×8 (160 ch)
Upsampling:
  - After TConv 1        16×16
  - After TConv 2        32×32
  - After TConv 3        64×64
  - After TConv 4        128×128
  - After TConv 5        256×256
Output                   256×256 (RGB)
```

## Channel Dimension Tracking

```
Stage                    Upper Branch    Lower Branch
─────────────────────────────────────────────────────
Input (from DWT)         3               3
Preprocessing            32              32
Branch Processing:
  - After 1st block      64              32
  - After 2nd block      64              32
  - After 3rd block      64              -
  - After 4th block      128             -
  - After 5th block      128             -
  - After 6th block      128             -
Alignment                128             32
Concatenation            160
Upsampling:
  - After Conv           128
  - After TConv 1        128
  - After TConv 2        64
  - After TConv 3        32
  - After TConv 4        16
  - After TConv 5        8
  - After Final Conv     3
```

## Memory Footprint Estimation

For a single 256×256 image (batch size = 1):

```
Stage                    Size            Memory (approx)
─────────────────────────────────────────────────────────
Input SAR                1×256×256       256 KB
After DWT                7×varying       ~200 KB
Upper branch:
  - Max intermediate     128×16×16       512 KB
Lower branch:
  - Max intermediate     32×64×64        512 KB
Concatenated             160×8×8         40 KB
Upsampling:
  - Max intermediate     128×16×16       512 KB
Output RGB               3×256×256       768 KB

Total Peak Memory: ~2-3 MB per image
```

## Comparison: Before vs After Fix

### Before Fix (Incorrect Implementation)

```
g2 (3×64×64) ──┐
               ├─> Concatenate ──> Process together
g3 (3×128×128)─┘
```

**Issues:**
- g2 and g3 processed jointly (not separately as in paper)
- No proper spatial alignment
- Incorrect dimension handling

### After Fix (Correct Implementation)

```
g2 (3×64×64) ──> Upper Branch ──┐
                                ├─> Align & Concatenate ──> Upconv
g3 (3×128×128) ─> Lower Branch ─┘
```

**Improvements:**
✓ Separate processing paths as per paper
✓ Proper spatial alignment before concatenation
✓ Correct upsampling from 8×8 to 256×256
✓ Architecture matches Figure 5 in CFRWD paper
