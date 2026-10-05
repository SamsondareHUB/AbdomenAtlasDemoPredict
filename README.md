readme_path = "/content/AbdomenAtlasDemoPredict/README.md"

readme_content = """# AbdomenAtlas Demo – 3D Vertebrae Segmentation (BodyMaps Warm-up)

This repository contains a reproducible demo for **3D vertebrae segmentation** on abdominal CT scans, implemented as a warm-up project for the **BodyMaps Research Assistant Program** at Johns Hopkins University. [1]

The pipeline:
- Loads a 3D CT volume (NIfTI format).
- Runs inference with a **SuPreM-style** 3D segmentation model (SwinUNETR backbone). [21]
- Splits the multi-class prediction into per-vertebra binary masks.
- Applies post-processing to clean the masks (largest-component filtering, morphological smoothing, hole filling).

The code is written in **PyTorch** and **MONAI**, and is structured to be easy to run, inspect, and extend.

---

## Requirements

- Python 3.9+
- PyTorch (CUDA recommended, but CPU works for small volumes)
- MONAI, nibabel, scipy, matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Place your CT volume (NIfTI, `.nii.gz`) in:

```text
data/sample_case/ct.nii.gz
```

You can rename the file, but update the command below accordingly.

> **Note:** The included `ct.nii.gz` is a small synthetic volume for demonstration only. For serious use or further experiments, replace it with a real abdominal CT scan.

---

## Model

Place the SuPreM vertebrae checkpoint in:

```text
weights/suprem_vertebrae_unet.pt
```

> **Note:** The included checkpoint is a randomly initialized placeholder. To use the official SuPreM weights, download them from the SuPreM repository and replace this file. [21]

The model architecture follows the SuPreM / AbdomenAtlas family of 3D medical imaging backbones developed at Johns Hopkins University. [21][19]

---

## Usage

### 1. Run segmentation

```bash
python predict.py \\
  --ct data/sample_case/ct.nii.gz \\
  --checkpoint weights/suprem_vertebrae_unet.pt \\
  --output outputs/sample_case
```

This creates per-vertebra masks:

```text
outputs/sample_case/vertebrae_01.nii.gz
...
outputs/sample_case/vertebrae_24.nii.gz
```

Each file corresponds to one vertebra (e.g., C1–L5 / sacral levels, depending on the class mapping).

### 2. Run post-processing

```bash
python postprocessing_vertebrae.py \\
  --input_dir outputs/sample_case \\
  --output_dir outputs/sample_case_postprocessed
```

This applies:

- **Largest-component filtering:** removes small disconnected regions.
- **Morphological smoothing:** opening and closing to clean boundaries.
- **Hole filling:** ensures solid vertebral body masks.

Cleaned masks are saved in `outputs/sample_case_postprocessed`.

---

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

- `predict.py`: 3D inference pipeline using a SuPreM-style SwinUNETR model. [21]
- `postprocessing_vertebrae.py`: post-processing utilities for vertebral masks.
- `configs/vertebrae_config.yaml`: basic configuration for model and inference parameters.

---

## Results

The demo produces 24 per-vertebra binary masks for a given CT volume. Example outputs (before and after post-processing) are stored in:

- `outputs/sample_case/` – raw predictions.
- `outputs/sample_case_postprocessed/` – cleaned masks.

You can open these NIfTI files in any medical imaging viewer (e.g., **3D Slicer**, **ITK-SNAP**, **FSLeyes**) to inspect the segmentations.

Typical post-processing effects:

- Removal of small isolated false-positive regions.
- Smoother vertebral boundaries.
- More anatomically coherent masks with fewer holes.

---

## Notes

- This project is intended as a **warm-up** for the BodyMaps Research Assistant Program. [1]
- The implementation is based on concepts from **SuPreM** and **AbdomenAtlas**, large-scale 3D medical imaging initiatives from Johns Hopkins University. [21][19]
- You can modify `configs/vertebrae_config.yaml` to experiment with different model depths, ROI sizes, and inference settings.

---

## Citation

If you use SuPreM or AbdomenAtlas in your work, please cite the corresponding papers and repositories:

- **SuPreM**: https://github.com/MrGiovanni/SuPreM [21]
- **AbdomenAtlas**: https://github.com/MrGiovanni/AbdomenAtlas [19]
- **BodyMaps**: https://www.thebodymaps.com [16]

---

## License

This demo is provided for educational and research purposes. Please respect the licenses and usage terms of the underlying datasets and pre-trained models (SuPreM, AbdomenAtlas) when using them in your own work.
"""

with open(readme_path, "w") as f:
    f.write(readme_content)

print("Updated README at:", readme_path)
!head -60 {readme_path}
