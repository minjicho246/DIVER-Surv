"""Dataset-specific image paths and survival labels."""

import os


def index_ucsf(
    df, img_dir, text_embeddings, required=("FLAIR_bias", "T1_bias", "T1c_bias", "T2_bias")
):
    """Build UCSF records using substring-based filename lookup."""
    items = []
    for (_, row), emb in zip(df.iterrows(), text_embeddings):
        pid = row["ID"]
        num = pid.split("-")[-1].zfill(4)
        folder = os.path.join(img_dir, f"UCSF-PDGM-{num}_nifti")
        paths = {}
        try:
            files = os.listdir(folder)
            for key in required:
                match = next((f for f in files if key in f), None)
                if match:
                    paths[key] = os.path.join(folder, match)
                else:
                    paths[key] = "MISSING"
                    print(f"missing patient {pid} file {key}")

        except OSError:
            paths = {k: "MISSING" for k in required}

        items.append(
            {
                "images": paths,
                "text_emb": emb,
                "event": int(row["1-dead 0-alive"]),
                "time": float(row["OS"]),
            }
        )
    return items


def index_upenn(
    df, img_dir, text_embeddings, required=("T1.nii.gz", "T1GD.nii.gz", "T2.nii.gz", "FLAIR.nii.gz")
):
    """Build UPENN records from patient IDs and modality filename suffixes."""
    items = []
    for (_, row), emb in zip(df.iterrows(), text_embeddings):
        pid = str(row["ID"])
        folder = os.path.join(img_dir, pid)
        paths = {key: os.path.join(folder, f"{pid}_{key}") for key in required}
        items.append(
            {
                "images": paths,
                "text_emb": emb,
                "event": int(row["event"]),
                "time": float(row["time"]),
            }
        )
    return items


def index_rhuh(
    df, img_dir, text_embeddings, required_modalities=("t1", "t1ce", "t2", "flair"), timepoint="0"
):
    """Build RHUH records from patient IDs and a fixed imaging timepoint."""
    items = []

    for (_, row), emb in zip(df.iterrows(), text_embeddings):
        pid = str(row["Patient ID"])

        folder = os.path.join(img_dir, pid, timepoint)

        paths = {}
        for mod in required_modalities:
            file_name = f"{pid}_{timepoint}_{mod}.nii.gz"
            full_path = os.path.join(folder, file_name)

            if os.path.exists(full_path):
                paths[mod] = full_path
            else:
                paths[mod] = "MISSING"
                print(f"missing patient {pid} file {mod}")

        items.append(
            {
                "images": paths,
                "text_emb": emb,
                "event": int(row["event"]),
                "time": float(row["Overall survival [OS] (days)"]),
            }
        )
    return items
