# SEM Microstructure Semantic Segmentation

**UNet (ResNet-34 Encoder) · PyTorch · Transfer Learning · Multi-class Pixel Segmentation**

A deep learning pipeline for automatic semantic segmentation of Scanning Electron Microscopy (SEM) images, identifying five distinct microstructural phases at pixel level — replacing manual annotation with a reproducible, production-ready AI system.

---

## Overview

This project develops a multi-class semantic segmentation system for SEM micrographs of additively manufactured Ni-WC metal matrix composites. The goal is to automatically label each pixel into one of five physical phases, enabling quantitative microstructure analysis without manual intervention.

The pipeline combines a UNet architecture with a pretrained ResNet-34 encoder, trained end-to-end using a hybrid loss function and evaluated with region-level, boundary-level, and class-level metrics. The work bridges **materials science and computer vision**, demonstrating practical deep learning for scientific microscopy.

---

## Dataset

SEM micrographs sourced from:

> Safdar, M. (2025). *Scanning Electron Microscopy (SEM) Dataset of Additively Manufactured Ni-WC Metal Matrix Composites for Semantic Segmentation* (Version 1). Zenodo. https://doi.org/10.5281/zenodo.17315241

| Property         | Detail                          |
|------------------|---------------------------------|
| Total images     | 405 SEM micrographs             |
| Resolution       | 512 × 512 pixels                |
| Input channels   | 1 (grayscale)                   |
| Segmentation classes | 5 microstructural phases    |
| Annotation type  | Pixel-wise masks                |

### Segmentation Classes

| Class ID | Phase          |
|----------|----------------|
| 0        | Matrix         |
| 1        | Carbide        |
| 2        | Void           |
| 3        | Reprecipitate  |
| 4        | Dilution Zone  |

### Data Split

Group-aware splitting was used to prevent data leakage between augmented variants of the same base image.

| Split      | Images |
|------------|--------|
| Train      | 280    |
| Validation | 71     |
| Test       | 54     |

---

## Model Architecture

Built with [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch):

```python
cfg = PretrainedUNetConfig(
    encoder_name    = "resnet34",
    encoder_weights = "imagenet",
    in_channels     = 1,
    num_classes     = 5,
)
model = build_smp_unet(cfg).to(device)
```

The encoder was initialised with ImageNet weights and adapted for single-channel grayscale SEM input.

### Training Strategy

| Stage   | Description                                          |
|---------|------------------------------------------------------|
| Stage 1 | Encoder frozen — decoder trained from scratch        |
| Stage 2 | Full network unfrozen — end-to-end fine-tuning        |
| Optimizer | AdamW with weight decay                            |
| Scheduler | ReduceLROnPlateau (monitors validation loss)       |
| Early stopping | Validation mIoU with configurable patience  |

---

## Loss Function

A hybrid loss was used to handle class imbalance and improve boundary precision:

```
Loss = 0.6 × CrossEntropyLoss(weighted) + 0.4 × DiceLoss
```

Weighted CrossEntropy penalises errors on underrepresented classes (Void, Carbide). Dice Loss directly optimises region overlap, complementing the pixel-wise CE signal.

---

## Evaluation Metrics

| Metric          | What it measures                                          |
|-----------------|-----------------------------------------------------------|
| mIoU            | Mean Intersection over Union — region overlap quality     |
| Dice Score      | Harmonic mean of precision and recall over segmented regions |
| Boundary F1     | Edge alignment precision with tolerance radius            |
| Confusion Matrix| Per-class error breakdown at pixel level                  |

IoU and Dice are computed from a **global confusion matrix** accumulated across the full test set, avoiding bias from unequal batch sizes.

---

## Results

Evaluated on 54 held-out test images, scored at the **image level** to avoid batch-size bias.

### Aggregate Test Performance

| Metric | Mean  | Std   | Min   | Max   |
|--------|-------|-------|-------|-------|
| mIoU   | 0.872 | 0.088 | 0.723 | 0.958 |
| mDice  | 0.912 | 0.079 | 0.759 | 0.978 |
| mBF1   | 0.728 | 0.027 | 0.678 | 0.773 |

### Per-Class Performance (from Global Confusion Matrix)

| Class         | IoU   | Dice  | Correctly Classified (%) |
|---------------|-------|-------|--------------------------|
| Matrix        | 0.939 | 0.969 | 97.3                     |
| Carbide       | 0.753 | 0.859 | 75.4                     |
| Void          | 0.976 | 0.988 | 98.9                     |
| Reprecipitate | 0.891 | 0.942 | 94.6                     |
| Dilution Zone | 0.881 | 0.937 | 93.3                     |

The **Void** and **Matrix** classes achieved near-perfect classification. The **Carbide** class was the most challenging, with 22.3% of Carbide pixels misclassified as Matrix — attributable to the visual similarity between fine carbide precipitates and the surrounding matrix at certain imaging conditions.

### Confusion Matrix (row-normalised %)

```
              Matrix  Carbide   Void  Reprecip  Dilution
Matrix         97.26     0.01   0.00      0.76      1.97
Carbide        22.30    75.38   0.01      0.07      2.25
Void            0.00     0.00  98.86      1.14      0.00
Reprecipitate   1.91     0.00   2.78     94.63      0.68
Dilution        4.84     0.00   0.02      1.87     93.26
```

---

## Boundary Quality

Microstructure analysis depends critically on **phase interfaces**, not just region coverage. Boundary F1 was computed using a dilation tolerance radius of 2 pixels to account for sub-pixel annotation uncertainty.

An mBF1 of **0.728** indicates the model produces well-aligned boundaries across most classes. The lower boundary scores relative to IoU and Dice are consistent with the fine structural detail and low inter-phase contrast inherent to SEM imaging.

---

## Project Structure
```
sem_image_segmentation/
├── model.py          # UNetScratch and pretrained SMP UNet
├── train.py          # Training loop, loss, scheduler, optimizer
├── evaluate.py       # Checkpoint loading, evaluation, test pipeline
├── metrics.py        # IoU, Dice, confusion matrix, boundary F1
├── dataset.py        # SEMSegDataset with group-aware splitting
├── utils.py          # Prediction overlays and metric visualisations
│
├── notebooks/
│   └── sem_image_segmentation_UNet.ipynb
│
├── checkpoints/
│   ├── best_model.pth          # Best pretrained encoder model (ResNet-34)
│   └── ScratchUNet_best_unet.pth   # Best scratch-built UNet model
│
└── images/           # Confusion matrix plots, overlay visualisations
```
## Download model checkpoints
    https://huggingface.co/imranlabs/sem-microstructure-segmentation/tree/main
---

## Tech Stack

- Python 3.12
- PyTorch
- segmentation-models-pytorch
- NumPy
- Matplotlib / Seaborn
- Pillow

---

## Limitations and Future Work

The Carbide class presents a persistent source of confusion with the Matrix class. Potential improvements include targeted augmentation for underrepresented phases, harder negative mining, or boundary-aware loss functions. The relatively modest mBF1 scores suggest post-processing refinement (e.g. conditional random fields) could improve edge precision.

Planned next steps:

- Deploy as a REST inference API using **FastAPI + Docker**
- Explore **SAM / foundation models** for annotation assistance
- Extend to **instance segmentation** for particle-level analysis
- Add **uncertainty estimation** for metrology reliability

---

## Why This Matters

Semantic segmentation of SEM images enables automated microstructure quantification, faster materials characterisation, AI-assisted metrology, and significant reduction in manual labelling effort. This project demonstrates that deep learning techniques developed for natural images can be successfully adapted to scientific microscopy with domain-specific modifications to data handling, loss design, and evaluation.

---

## Author

**Imran Khan**
Physics PhD · SEM Metrology Engineer · Computer Vision / AI
