
import os
import argparse
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 25  # background + 24 vertebrae (adjust if your model uses a different number)


def load_vertebrae_model(checkpoint_path, device=DEVICE):
    """
    Load a SuPreM-style vertebrae segmentation model.
    Adjust architecture/hyperparameters to match the checkpoint you are using.
    """
    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=48,
        img_size=(96, 96, 96),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

    state = torch.load(checkpoint_path, map_location=device)

    # Handle common checkpoint formats
    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_ct(ct_path):
    """
    Load a NIfTI CT and apply simple intensity normalization.
    Refine this to match the preprocessing used when training your model.
    """
    img = nib.load(ct_path)
    data = img.get_fdata().astype(np.float32)

    # Simple z-score normalization; replace if your model expects different scaling
    mean, std = data.mean(), data.std() + 1e-8
    data = (data - mean) / std

    # Add batch and channel dims: (B, C, H, W, D)
    data = data[None, None]
    return data, img.affine, img.header


@torch.no_grad()
def predict_vertebrae(model, ct_path, output_dir, roi_size=(96, 96, 96), device=DEVICE):
    """
    Run sliding-window inference and save per-vertebra NIfTI masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    data, affine, header = preprocess_ct(ct_path)
    data = torch.from_numpy(data).to(device)

    # Sliding window inference for large 3D volumes
    logits = sliding_window_inference(
        inputs=data,
        roi_size=roi_size,
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
        mode="gaussian",
    )  # (B, num_classes, H, W, D)

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).cpu().numpy()[0]  # (H, W, D)

    num_classes = pred.max() + 1
    print(f"Predicted classes: 0..{num_classes - 1}")

    # Save one binary mask per vertebra (skip background class 0)
    for c in range(1, num_classes):
        mask = (pred == c).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask, affine, header)
        out_name = f"vertebrae_{c:02d}.nii.gz"
        out_path = os.path.join(output_dir, out_name)
        nib.save(mask_img, out_path)
        print("Saved:", out_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True, help="Path to input CT (NIfTI .nii.gz)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output directory for per-vertebra masks")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96],
                        help="ROI size for sliding window inference")
    parser.add_argument("--device", default=DEVICE, help="Device (cuda/cpu)")
    args = parser.parse_args()

    model = load_vertebrae_model(args.checkpoint, device=args.device)
    predict_vertebrae(
        model,
        args.ct,
        args.output,
        roi_size=tuple(args.roi_size),
        device=args.device,
    )


if __name__ == "__main__":
    main()
