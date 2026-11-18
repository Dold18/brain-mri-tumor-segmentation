"""Interactive, checkpoint-backed MRI whole-tumor-region segmentation viewer.

Run: streamlit run app/streamlit_app.py
Requires an existing checkpoint, the official MSD dataset, split manifest and
preprocessed patient cache. No clinical diagnostic functionality is provided.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import streamlit as st

from brain_mri.cache import PreprocessSettings, case_directory, read_case_metadata
from brain_mri.data import read_catalog
from brain_mri.inference import load_checkpoint_model, predict_cached_case
from brain_mri.splits import validate_splits
from brain_mri.storage_paths import (
    BASELINE_RUN_DIR, CACHE_DIR, DATASET_DIR, QUICKSTART_CHECKPOINT, SPLIT_MANIFEST,
)
from brain_mri.train import _hash, select_device
from brain_mri.viewer import disagreement_overlay, foreground_slice_index, patient_metrics
from brain_mri.visualization import color_overlay, grayscale_uint8, save_case_comparison


st.set_page_config(
    page_title="Brain MRI Tumor Segmentation | Research Viewer",
    page_icon="🧠", layout="wide", initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {max-width: 1600px; padding-top: 1.7rem;}
[data-testid="stMetric"] {padding: .6rem 0;}
</style>
""", unsafe_allow_html=True)

st.title("Brain MRI Tumor Segmentation")
st.caption(
    "Four-sequence MRI · 2D U-Net · patient-volume inference  |  "
    "Research software, not validated for clinical use or diagnosis"
)

baseline_checkpoint = BASELINE_RUN_DIR / "best.pt"
default_checkpoint = baseline_checkpoint if baseline_checkpoint.is_file() else QUICKSTART_CHECKPOINT

with st.sidebar:
    st.header("Explore a study")
    st.caption("Predictions are generated locally by the trained neural network.")
    with st.expander("Advanced model and data paths", expanded=False):
        checkpoint_path = Path(st.text_input("Trained checkpoint", str(default_checkpoint)))
        dataset_path = Path(st.text_input("MSD dataset", str(DATASET_DIR)))
        split_path = Path(st.text_input("Patient split manifest", str(SPLIT_MANIFEST)))
        cache_root = Path(st.text_input("Preprocessed MRI cache", str(CACHE_DIR)))
        device_choice = st.selectbox("Inference device", ["auto", "cpu", "mps", "cuda"])


@st.cache_resource(show_spinner="Loading trained MRI model...")
def get_model(path: str, timestamp: int, device: str):
    del timestamp  # part of cache key; replaced checkpoints must not use stale weights
    return load_checkpoint_model(path, device=device)


@st.cache_data(show_spinner="Running inference on the full MRI study...")
def get_prediction(case_id: str, cache_path: str, checkpoint_path: str,
                   checkpoint_stamp: int, image_stamp: int, device: str) -> np.ndarray:
    del image_stamp  # used for cache invalidation if source MRI cache changes
    model, _ = get_model(checkpoint_path, checkpoint_stamp, device)
    image_slices = np.load(case_directory(cache_path, case_id) / "image.npy", mmap_mode="r")
    return predict_cached_case(model, image_slices, batch_size=8, device=device)


if not checkpoint_path.is_file():
    st.info("No checkpoint found. Set the path to your trained `runs/baseline/best.pt` in the sidebar.")
    st.stop()
if not dataset_path.is_dir() or not split_path.is_file():
    st.info("The MSD dataset and saved patient-level split manifest are required to open the viewer.")
    st.stop()

try:
    device = str(select_device(device_choice))
    checkpoint_stamp = checkpoint_path.stat().st_mtime_ns
    _, checkpoint = get_model(str(checkpoint_path), checkpoint_stamp, device)
    catalog = read_catalog(dataset_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    lookup = {record.case_id: record for record in catalog.cases}
    validate_splits(splits, set(lookup))
    if checkpoint["provenance"]["split_hash"] != _hash(splits):
        raise ValueError("Checkpoint and patient split manifest do not match")
    cfg = checkpoint["preprocessing"]
    settings = PreprocessSettings(
        float(cfg["clip_low_percentile"]), float(cfg["clip_high_percentile"]),
        int(checkpoint["model_settings"]["in_channels"]),
    )
    threshold = float(checkpoint["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Checkpoint contains an invalid decision threshold")
except (ValueError, RuntimeError, FileNotFoundError, KeyError, OSError) as exc:
    st.error(f"Unable to load the project: {exc}")
    st.stop()

with st.sidebar:
    group = st.selectbox("Dataset partition", ("validation", "test"))
    if group == "test":
        st.caption("This is the already-evaluated held-out partition. Do not tune the model using these scans.")
    available_ids = [cid for cid in splits[group]
                     if (case_directory(cache_root, cid) / "meta.json").is_file()]
    if not available_ids:
        st.warning("No cached MRI studies in this partition. Run `scripts.prepare_cache`.")
        st.stop()
    case_id = st.selectbox("Patient study", available_ids)

try:
    read_case_metadata(cache_root, lookup[case_id], settings)
    folder = case_directory(cache_root, case_id)
    image_path = folder / "image.npy"
    image_stamp = image_path.stat().st_mtime_ns
    images = np.load(image_path, mmap_mode="r")
    masks = np.load(folder / "mask.npy", mmap_mode="r")
    if masks.shape != (len(images), 1, *images.shape[-2:]):
        raise ValueError("Image and annotation caches have incompatible shapes")
except (OSError, FileNotFoundError, ValueError) as exc:
    st.error(f"Unable to load this MRI study: {exc}")
    st.stop()

names = [catalog.modalities.get(str(i), f"Modality {i}") for i in range(images.shape[1])]
with st.sidebar:
    modality = st.selectbox("MRI sequence", list(range(len(names))),
                            format_func=lambda i: names[i])
    truth_volume = np.asarray(masks[:, 0], dtype=bool)
    default_slice = foreground_slice_index(truth_volume)
    slice_index = st.slider("Axial slice", 0, len(images) - 1, default_slice,
                            key=f"slice_{case_id}")
    opacity = st.slider("Overlay opacity", 0.10, 0.90, 0.50, 0.05)
    inference_key = (
        f"{checkpoint_path.resolve()}::{checkpoint_stamp}::{cache_root.resolve()}::"
        f"{case_id}::{image_stamp}::{device}"
    )
    if st.button("Run whole-volume inference", type="primary", width="stretch"):
        st.session_state["active_inference_key"] = inference_key
    st.divider()
    st.caption(f"Checkpoint epoch {checkpoint['epoch']} · Fixed threshold {threshold:.2f}")
    st.caption(f"{len(images)} axial slices · {len(names)} MRI modalities · {device.upper()}")
    if checkpoint["provenance"].get("exploratory_subset", True):
        st.warning("Exploratory-subset checkpoint. Do not present its output as final model performance.")

active = st.session_state.get("active_inference_key") == inference_key
probabilities = None
binary_prediction = None
if active:
    try:
        probabilities = get_prediction(
            case_id, str(cache_root), str(checkpoint_path), checkpoint_stamp,
            image_stamp, device,
        )
        if probabilities.shape != truth_volume.shape:
            raise ValueError("Model prediction and annotation volumes are not aligned")
        binary_prediction = probabilities >= threshold
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Model inference failed: {exc}")
        st.stop()

st.subheader(f"Study {case_id}")
st.caption(
    f"Axial slice {slice_index + 1} of {len(images)} · {names[modality]} sequence · "
    "annotations available for research comparison"
)

slice_img = np.asarray(images[slice_index, modality], dtype=np.float32)
slice_truth = truth_volume[slice_index]
panels = st.columns(3, gap="medium")
with panels[0]:
    st.image(grayscale_uint8(slice_img), caption="Original MRI", width="stretch")
with panels[1]:
    if active and binary_prediction is not None:
        st.image(color_overlay(slice_img, binary_prediction[slice_index], alpha=opacity),
                 caption="Model prediction (coral)", width="stretch")
    else:
        st.info("Run inference from the sidebar to view a model-generated segmentation.")
with panels[2]:
    st.image(color_overlay(slice_img, slice_truth, color=(40, 205, 225), alpha=opacity),
             caption="Ground-truth reference (cyan)", width="stretch")

if active and probabilities is not None and binary_prediction is not None:
    results = patient_metrics(probabilities, truth_volume, threshold=threshold)
    st.caption("Metrics below are computed over the **entire patient volume**, not just the current slice.")
    cols = st.columns(4)
    for col, label, key in zip(
        cols, ("Dice", "Intersection over union", "Precision", "Recall"),
        ("dice", "iou", "precision", "recall"),
    ):
        col.metric(label, f"{results[key]:.3f}")

    with st.expander("Inspect prediction errors", expanded=False):
        left, right = st.columns([1.15, 1])
        with left:
            st.image(
                disagreement_overlay(slice_img, binary_prediction[slice_index],
                                     slice_truth, alpha=opacity),
                caption="Agreement and errors on selected axial slice", width="stretch",
            )
        with right:
            st.markdown(
                "**Color key**  \n"
                "Green: correctly predicted region  \n"
                "Red: predicted outside the reference  \n"
                "Blue: annotated region missed by the model"
            )
            st.write(f"False-positive voxels (full scan): **{results['false_positive_voxels']:,}**")
            st.write(f"False-negative voxels (full scan): **{results['false_negative_voxels']:,}**")
            st.caption("Errors on one selected slice do not represent the full-volume evaluation.")

    # Export a genuine prediction comparison, without embedding or modifying the checkpoint.
    png_buffer = BytesIO()
    save_case_comparison(
        png_buffer, slice_img, slice_truth, binary_prediction[slice_index],
        case_id=case_id, slice_index=slice_index,
    )
    st.download_button(
        "Download this slice comparison (PNG)", png_buffer.getvalue(),
        file_name=f"{case_id}_slice_{slice_index:03d}_comparison.png",
        mime="image/png",
    )

st.divider()
st.caption(
    "Model target: binary whole-tumor-associated region (including edema) from the MSD "
    "Task01_BrainTumour glioma dataset. All cases shown are from this dataset. "
    "No healthy-vs-tumor diagnostic validation, no external hospital validation, "
    "and no clinical approval. Canonical image orientation is not necessarily the "
    "radiological left/right viewing convention."
)
