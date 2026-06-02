from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


def patient_id_from_filename(name: str) -> str:
    return (
        name
        .replace("_CT.nii.gz", "")
        .replace("_PET.nii.gz", "")
        .replace("_CT.nii", "")
        .replace("_PET.nii", "")
    )


def compute_stats(split_dir: str):
    split_dir = Path(split_dir)

    for modality in ["CT", "PET"]:
        rows = []
        files = sorted([
            p for p in split_dir.iterdir()
            if p.name.endswith((".nii", ".nii.gz")) and f"_{modality}" in p.name
        ])

        print(f"{split_dir} | {modality}: {len(files)} files")

        for f in files:
            arr = nib.load(str(f)).get_fdata().astype(np.float32)

            rows.append({
                "name": patient_id_from_filename(f.name),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "99_percentile": float(np.percentile(arr, 99)),
            })

        out = split_dir / f"{modality.lower()}_min_max_percentile.csv"
        pd.DataFrame(rows, columns=["name", "min", "max", "99_percentile"]).to_csv(out, index=False)
        print(f"saved: {out}")


base = Path("../data/PT_CT_BAT")

for split in ["train", "val", "test"]:
    compute_stats(base / split)