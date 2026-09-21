# Dataset preparation

Provide the three metadata CSV files and corresponding NIfTI image directories.
The repository contains no patient records, generated summaries, scans, or weights.
The loader assumes that the metadata is already prepared; it does not generate
patient summaries, filter rows, harmonize survival units, or impute missing values.

## Required columns

| Cohort | Patient identifier | Event | Survival time | Text |
| --- | --- | --- | --- | --- |
| UCSF | `ID` | `1-dead 0-alive` | `OS` | `Patient_Summary` |
| UPENN | `ID` | `event` | `time` | `Patient_Summary` |
| RHUH | `Patient ID` | `event` | `Overall survival [OS] (days)` | `Patient_Summary` |

Event values are converted to integers, with `1` indicating an observed event and
`0` censoring. Times are converted to floats without unit conversion. Keep time
units consistent when comparing cohorts. Patient summaries are passed directly to
the configured `SentenceTransformer`.

## Image directory conventions

The examples below describe naming patterns only.

### UCSF

```text
<ucsf_img_dir>/
└── UCSF-PDGM-0001_nifti/
    ├── <filename containing T1_bias>
    ├── <filename containing T1c_bias>
    ├── <filename containing T2_bias>
    └── <filename containing FLAIR_bias>
```

The final component of the metadata `ID`, split on `-`, is zero-padded to four
characters. For each required key, the first matching entry returned by
`os.listdir` is used; no filename sorting or new disambiguation rule was added.

### UPENN

```text
<upenn_img_dir>/
└── <ID>/
    ├── <ID>_T1.nii.gz
    ├── <ID>_T1GD.nii.gz
    ├── <ID>_T2.nii.gz
    └── <ID>_FLAIR.nii.gz
```

### RHUH

```text
<rhuh_img_dir>/
└── <Patient ID>/
    └── 0/
        ├── <Patient ID>_0_t1.nii.gz
        ├── <Patient ID>_0_t1ce.nii.gz
        ├── <Patient ID>_0_t2.nii.gz
        └── <Patient ID>_0_flair.nii.gz
```

The experiment uses RHUH timepoint `0`.

## Preprocessing and channel order

All cohorts enter the model in the order **T1, contrast-enhanced T1, T2, FLAIR**.
The modality key order in `experiment.py` controls concatenation, independently of
the dictionary insertion order returned by the indexers.

The deterministic pipeline loads NIfTI images with `NibabelReader`, ensures channel
first tensors, resizes each modality to `(120, 120, 64)` using trilinear
interpolation, scales intensities to `[0, 1]`, and concatenates four channels.
MONAI `PersistentDataset` caches this preprocessing on disk.

Training applies independent random flips along spatial axes `0` and `1`, each
with probability `0.5`, after reading the cached item. Evaluation has no random
flips. Final items contain `image`, `text_emb`, `event`, and `time`.

## Existing input assumptions

Missing UCSF/RHUH paths may be recorded as `"MISSING"`, and UPENN paths are
constructed without existence checks. These records are not automatically skipped
or repaired; image loading can fail. The full-input experiment expects all four
modalities. The current training entry point requires all four modalities.

Embeddings follow the CSV row order. Source subsets retain the dataframe
index-based tensor selection. Metadata loaded with `pd.read_csv` therefore uses
the same default row index as in the source script.
