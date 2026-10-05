# AbdomenAtlas Demo – Vertebrae Segmentation (BodyMaps Warm-up)

This demo runs 3D vertebrae segmentation on a CT scan using a SuPreM-style model, then applies post-processing to clean the masks. It is intended as a warm-up project for the **BodyMaps Research Assistant Program** at Johns Hopkins University. [1]

## Requirements

- Python 3.9+
- PyTorch with CUDA (or CPU)
- MONAI, nibabel, scipy, matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place your CT volume (NIfTI, `.nii.gz`) in:

```text
data/sample_case/ct.nii.gz
```

You can rename the file, but update the command below accordingly.

> Note: The included `ct.nii.gz` is a small synthetic volume for demonstration only. For serious use, replace it with a real abdominal CT.

## Model

Place the SuPreM vertebrae checkpoint in:

```text
weights/suprem_vertebrae_unet.pt
```

> Note: The included checkpoint is a randomly initialized placeholder. To use the official SuPreM weights, download them from the SuPreM repository and replace this file. [21]

## Usage

### 1. Run segmentation

```bash
python predict.py \
  --ct data/sample_case/ct.nii.gz \
  --checkpoint weights/suprem_vertebrae_unet.pt \
  --output outputs/sample_case
```

This creates per-vertebra masks:

```text
outputs/sample_case/vertebrae_01.nii.gz
...
outputs/sample_case/vertebrae_24.nii.gz
```

### 2. Run post-processing

```bash
python postprocessing_vertebrae.py \
  --input_dir outputs/sample_case \
  --output_dir outputs/sample_case_postprocessed
```

This applies:

- Largest-component filtering
- Morphological smoothing
- Hole filling

Cleaned masks are saved in `outputs/sample_case_postprocessed`.

## Project structure

```text
AbdomenAtlasDemoPredict/
├── README.md
├── requirements.txt
├── predict.py
├── postprocessing_vertebrae.py
├── configs/
│   └── vertebrae_config.yaml
├── weights/
│   └── suprem_vertebrae_unet.pt
├── data/
│   └── sample_case/
│       └── ct.nii.gz
└── outputs/
    ├── sample_case/
    │   └── vertebrae_*.nii.gz
    └── sample_case_postprocessed/
        └── vertebrae_*.nii.gz
```

## Notes

- The model architecture follows SuPreM / AbdomenAtlas work from Johns Hopkins University. [21][19]
- This demo is intended as a warm-up for the BodyMaps Research Assistant Program. [1]
- Modify `configs/vertebrae_config.yaml` to change model and inference parameters.

## Citation

If you use SuPreM or AbdomenAtlas in your work, please cite the corresponding papers and repositories:

- SuPreM: https://github.com/MrGiovanni/SuPreM [21]
- AbdomenAtlas: https://github.com/MrGiovanni/AbdomenAtlas [19]
- BodyMaps: https://www.thebodymaps.com [16]
