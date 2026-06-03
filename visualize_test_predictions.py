from pathlib import Path
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from models.attention_unet import AttentionUNet


def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def normalize_ct(ct, ct_min, ct_p99):
    return (ct - ct_min) / (ct_p99 - ct_min + 1e-8)


def normalize_pet(pet, pet_p99):
    return pet / (pet_p99 + 1e-8)


def denormalize_pet(pet_norm, pet_p99):
    return pet_norm * pet_p99


def get_patient_id_from_ct(ct_name):
    return (
        ct_name
        .replace("_CT.nii.gz", "")
        .replace("_CT.nii", "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="../data/PT_CT_BAT/test")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--patient", type=str, default=None, help="ex: 2008_PRE")
    parser.add_argument("--slice", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="./prediction_figures")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ct_files = sorted(test_dir.glob("*_CT.nii.gz"))
    if args.patient is not None:
        ct_files = [p for p in ct_files if get_patient_id_from_ct(p.name) == args.patient]

    if len(ct_files) == 0:
        raise RuntimeError(f"No CT files found for patient={args.patient}")

    # CSV stats
    import pandas as pd
    df_ct = pd.read_csv(test_dir / "ct_min_max_percentile.csv")
    df_pet = pd.read_csv(test_dir / "pet_min_max_percentile.csv")

    model = AttentionUNet(16)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for ct_path in ct_files:
        patient_id = get_patient_id_from_ct(ct_path.name)
        pet_path = test_dir / f"{patient_id}_PET.nii.gz"

        if not pet_path.exists():
            print(f"[WARNING] Missing PET for {patient_id}")
            continue

        ct = load_nifti(ct_path)
        pet = load_nifti(pet_path)

        row_ct = df_ct[df_ct["name"] == patient_id].iloc[0]
        row_pet = df_pet[df_pet["name"] == patient_id].iloc[0]

        ct_norm = normalize_ct(
            ct,
            ct_min=float(row_ct["min"]),
            ct_p99=float(row_ct["99_percentile"]),
        )

        pet_norm = normalize_pet(
            pet,
            pet_p99=float(row_pet["99_percentile"]),
        )

        if args.slice is None:
            slice_idx = ct.shape[2] // 2
        else:
            slice_idx = args.slice

        ct_slice = ct_norm[:, :, slice_idx]
        pet_slice = pet_norm[:, :, slice_idx]

        x = torch.from_numpy(ct_slice).float()[None, None].to(device)

        with torch.no_grad():
            pred_logits, _ = model(x)

        pred_norm = pred_logits.squeeze().cpu().numpy()

        pet_min = float(row_pet["min"])
        pet_max = float(row_pet["max"])

        print("pet_min =", pet_min)
        print("pet_max =", pet_max)

        print("pet_slice:",
            pet_slice.min(),
            pet_slice.max(),
            pet_slice.mean())

        print("pred_norm:",
            pred_norm.min(),
            pred_norm.max(),
            pred_norm.mean())
        pet_pred = pred_norm * (pet_max - pet_min) + pet_min
        pet_gt = pet_slice * (pet_max - pet_min) + pet_min

        print("target SUV:", pet_gt.min(), pet_gt.max(), pet_gt.mean())
        print("pred SUV  :", pet_pred.min(), pet_pred.max(), pet_pred.mean())
        print("pred logits norm:", pred_logits.min().item(), pred_logits.max().item(), pred_logits.mean().item())

        err = np.abs(pet_pred - pet_gt)

        vmax_pet = max(np.percentile(pet_gt, 99), 1e-6)
        pet_gt_vis = np.clip(pet_gt / vmax_pet, 0.0, 1.0)
        pet_pred_vis = np.clip(pet_pred / vmax_pet, 0.0, 1.0)

        err_vmax = max(np.percentile(err, 99), 1e-6)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(ct[:, :, slice_idx], cmap="gray")
        axes[0].set_title(f"CT\n{patient_id} slice {slice_idx}")
        axes[0].axis("off")

        im1 = axes[1].imshow(pet_gt_vis, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"PET GT\nvis / p99 GT={vmax_pet:.2f} SUV")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(pet_pred_vis, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("PET pred\nsame scale as GT")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        im3 = axes[3].imshow(err, cmap="magma", vmin=0, vmax=err_vmax)
        axes[3].set_title(f"|pred - GT|\np99={err_vmax:.2f} SUV")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        plt.tight_layout()

        out_path = out_dir / f"{patient_id}_slice{slice_idx:03d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()