
import os
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_opening, binary_closing, binary_fill_holes


def keep_largest_component(mask): 
    """
    Keep only the largest connected component in a binary mask.
    """
    labeled, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    largest = np.argmax(sizes[1:]) + 1
    return (labeled == largest).astype(np.uint8)


def postprocess_vertebra_mask(mask):
    """
    Apply basic post-processing to a single vertebra mask:
    - Keep largest component
    - Morphological smoothing
    - Hole filling
    """
    if mask.sum() == 0:
        return mask

    # 1. Keep largest connected component
    mask = keep_largest_component(mask)

    # 2. Morphological smoothing
    mask = binary_opening(mask, structure=np.ones((3, 3, 3)))
    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))

    # 3. Fill holes
    mask = binary_fill_holes(mask)

    return mask.astype(np.uint8)


def postprocess_all_vertebrae(input_dir, output_dir):
    """
    Post-process all vertebrae_*.nii.gz files in input_dir and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(input_dir)
         if f.startswith("vertebrae_") and f.endswith(".nii.gz")]
    )

    if not files:
        print("No vertebrae_*.nii.gz files found in", input_dir)
        return

    for f in files:
        in_path = os.path.join(input_dir, f)
        img = nib.load(in_path)
        data = img.get_fdata().astype(np.uint8)

        data_clean = postprocess_vertebra_mask(data)

        out_img = nib.Nifti1Image(data_clean, img.affine, img.header)
        out_path = os.path.join(output_dir, f)
        nib.save(out_img, out_path)
        print("Postprocessed:", f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing vertebrae_*.nii.gz files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save post-processed masks")
    args = parser.parse_args()

    postprocess_all_vertebrae(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
