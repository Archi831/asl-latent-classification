# Visual Analysis Checklist — ayuraj

Refer to `dataset_analysis/samples_{name}.png` while filling this in.

## Quantitative Summary

    Dataset:          ayuraj
    Kaggle ID:        ayuraj/asl-dataset
    Total images:     2515
    Classes present:  36 / 39
    Missing classes:  ['nothing', 'space', 'unknown']
    Samples/class:    min=65  max=70  avg=70
    Native resol.:    [(400, 400)]
    Channel mode:     3 (RGB)
    Pixel mean (gs):  0.1733  (train: 0.5550)
    Pixel std  (gs):  0.0460  (train: 0.2380)
    Domain distance:  |mean diff| = 0.3817
    
    Per-class sample counts:
               0:    70  #######
               1:    70  #######
               2:    70  #######
               3:    70  #######
               4:    70  #######
               5:    70  #######
               6:    70  #######
               7:    70  #######
               8:    70  #######
               9:    70  #######
               A:    70  #######
               B:    70  #######
               C:    70  #######
               D:    70  #######
               E:    70  #######
               F:    70  #######
               G:    70  #######
               H:    70  #######
               I:    70  #######
               J:    70  #######
               K:    70  #######
               L:    70  #######
               M:    70  #######
               N:    70  #######
               O:    70  #######
               P:    70  #######
               Q:    70  #######
               R:    70  #######
               S:    70  #######
               T:    65  ######
               U:    70  #######
               V:    70  #######
               W:    70  #######
               X:    70  #######
               Y:    70  #######
               Z:    70  #######
         nothing:     0    ** MISSING **
           space:     0    ** MISSING **
         unknown:     0    ** MISSING **

## Visual Inspection

For each item, mark one option and add notes if needed.

| Aspect | Assessment | Notes |
|---|---|---|
| Background | [x] plain/uniform |black background(png)|
| Hand orientation | [x] varied | |
| Lighting | [x] controlled | |
| Skin tone diversity | [x] low | |
| Image quality | [x] clean | |
| Channel format | [x] RGB | |

## Problem Classes

List any classes that look visually incompatible with the ASL39 training data:

for the most part they are same to danrasband with the exception of: 
- D: completely different viewpoint, 
- G, H, Q, Z: (tilted but same viewpoint), 
- M, N, X: slightly different viewpoint
debashishsau: 
- D: completely different viewpoint (side view vs. front)
- G, H, J, Q: tilted but same viewpoint
- M, N: slightly different viewpoint (vs. mostly upright in training set)

- 

