# TSCG-TSM

Temporal-Spatial-Channel-Global enhanced TSM (TSCG-TSM) for fish feeding motivation evaluation.

## Overview

TSCG-TSM is an improved video action recognition model based on TSM, enhanced with three key modules:

- **SCGM (Spatial-Channel Gated Module)**: A dual-attention mechanism combining channel attention and spatial attention with gated feature modulation.
- **TAM (Temporal Attention Module)**: A temporal attention module to dynamically adjust the importance of different frames according to video content.
- **GRN (Global Response Normalization)**: A feature normalization technique from ConvNeXtV2 that improves feature diversity.

The model classifies fish feeding motivation into 4 levels: None (0), Weak (1), Medium (2), Strong (3).

## Dataset

The demo dataset used in this project is available at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20595836.svg)](https://doi.org/10.5281/zenodo.20595836)

Download the dataset and place the video files in `tools/data/`.

## Project Structure

```
TSCG-TSM/
├── configs/
│   └── TSCG-TSM.py          # Model and training configuration
├── modules/
│   ├── SCGM.py              # Spatial-Channel Gated Module
│   └── GRN.py               # Global Response Normalization
├── mmaction/                
│   ├── models/
│   │   ├── backbones/
│   │   │   └── TSCG_TSM.py  # TSCG-TSM backbone
│   │   └── heads/
│   │       └── tsm_head.py             # TSM head with temporal attention
│   ├── datasets/
│   ├── engine/
│   └── ...
├── tools/
│   ├── data/                # Dataset annotations
│   │   ├── trainlist.txt
│   │   ├── testlist.txt
│   │   └── classInd.txt
│   ├── train.py
│   └── test.py
├── weights/
│   └── best.pth             # best weights
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
python tools/test.py configs/TSCG-TSM.py weights/best.pth
```

## Training

```bash
python tools/train.py configs/TSCG-TSM.py
```

## Acknowledgments

- This project is built upon [MMAction2](https://github.com/open-mmlab/mmaction2). We thank the OpenMMLab team for their excellent framework.
