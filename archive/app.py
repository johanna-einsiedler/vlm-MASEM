#!/usr/bin/env python3
"""Streamlit app for reviewing extraction evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Extraction Evaluation Viewer", layout="wide")


# -------- Helper Functions --------

def _available_datasets() -> list[str]:
    """Return sorted list of dataset names that contain at least one evaluation JSON."""
    eval_root = Path("data/evaluation")
    if not eval_root.exists():
        return []
    return sorted(
        d.name for d in eval_root.iterdir()
        if d.is_dir() and any(d.glob("*.json"))
    )


def list_evaluation_files(evaluation_dir: Path) -> Dict[str, Path]:
    """List all evaluation JSON files."""
    if not evaluation_dir.exists():
        return {}
    files = sorted(evaluation_dir.glob("*.json"))
    return {path.stem: path for path in files}


def load_evaluation(path: Path) -> dict:
    """Load evaluation JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def get_study_images(study_base: str, images_root: Path) -> Dict[int, Path]:
    """Get all page images for a study."""
    study = study_base.rstrip("abc")
    study_dir = images_root / study
    if not study_dir.exists():
        return {}

    images = {}
    for img_path in study_dir.glob("page*.png"):
        try:
            page_num = int(img_path.stem.replace("page", ""))
            images[page_num] = img_path
        except ValueError:
            continue
    return images


def get_source_page(study_base: str, extraction_dir: Path) -> Optional[int]:
    """Read the best available page number from a 1-step extraction JSON file.

    Preference order:
    1. evidence.page from the first sample inside the result JSON (most precise)
    2. pages[0]["number"] from the outer wrapper (source_page from GPT suffix)
    """
    study = study_base.rstrip("abc")
    json_path = extraction_dir / f"{study}.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pages = data.get("pages", [])
        if not pages:
            return None

        # Try evidence.page inside the result JSON
        result_str = pages[0].get("result", "")
        if result_str:
            try:
                result = json.loads(result_str)
                samples = result.get("samples", [])
                if samples:
                    ev_page = samples[0].get("evidence", {}).get("page")
                    if ev_page is not None:
                        return int(ev_page)
            except Exception:
                pass

        # Fall back to outer wrapper page number
        return pages[0].get("number")
    except Exception:
        return None


def get_metadata_source_page(study_base: str, extraction_dir: Path) -> Optional[int]:
    """Read the most common evidence page cited across metadata fields."""
    study = study_base.rstrip("abc")
    json_path = extraction_dir / f"{study}.json"
    if not json_path.exists():
        return None
    try:
        result_str = json.loads(json_path.read_text(encoding="utf-8"))
        # metadata is saved as a raw string (not wrapped)
        if isinstance(result_str, str):
            result = json.loads(result_str)
        else:
            result = result_str
        records = result.get("records", [])
        if not records:
            return None
        evidence = records[0].get("evidence", {})
        pages = [
            v.get("page") for v in evidence.values()
            if isinstance(v, dict) and v.get("page") is not None
        ]
        if not pages:
            return None
        # Return the most common page; on tie, return the smallest
        return min(pages, key=lambda p: (-pages.count(p), p))
    except Exception:
        return None


def format_comparison_table(data: dict, data_type: str) -> pd.DataFrame:
    """Format extracted vs true values as a DataFrame."""
    rows = []
    for key, values in data.items():
        if not isinstance(values, dict):
            continue

        extracted = values.get("extracted")
        true = values.get("true")
        accuracy = values.get("accuracy", 0)

        if extracted is None:
            extracted_str = "null"
        elif isinstance(extracted, (int, float)):
            extracted_str = f"{extracted:.2f}" if extracted != 0 else "0.00"
        else:
            extracted_str = str(extracted)

        if true is None:
            true_str = "null"
        elif isinstance(true, (int, float)):
            true_str = f"{true:.2f}" if true != 0 else "0.00"
        else:
            true_str = str(true)

        match = "✓" if accuracy == 1 else "✗"

        rows.append({
            "Field": key,
            "Extracted": extracted_str,
            "True": true_str,
            "Match": match
        })

    return pd.DataFrame(rows)


def render_source_page_image(source_page: Optional[int], study_images: Dict[int, Path], label: str) -> None:
    """Display the source page image for an extraction result."""
    if source_page is not None and source_page in study_images:
        st.caption(f"{label} source page: {source_page}")
        img = Image.open(study_images[source_page])
        st.image(img, caption=f"Page {source_page}", use_container_width=True)
    elif source_page is None:
        st.caption("Source page not recorded for this study.")
    else:
        st.caption(f"Source page {source_page} — image not available (run pdf_to_images.py).")


def highlight_match(row):
    if row["Match"] == "✓":
        return ["background-color: #e8f5e9"] * len(row)
    else:
        return ["background-color: #ffebee"] * len(row)


# -------- Main App --------
st.title("Extraction Evaluation Viewer")

# -------- Dataset selection --------
available_datasets = _available_datasets()
if not available_datasets:
    st.error("No datasets found in data/evaluation/. Run the pipeline first.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Dataset", available_datasets, index=0)

EVALUATION_DIR = Path("data/evaluation") / selected_dataset
# Images are stored without the _1step suffix (same base dataset)
IMAGES_ROOT = Path("data/image_data") / selected_dataset.removesuffix("_1step")
EXTRACTION_FACTORS_DIR = Path("data/extraction_factors") / selected_dataset
EXTRACTION_CORRELATIONS_DIR = Path("data/extraction_correlations") / selected_dataset

# Load all evaluation files
evaluations = list_evaluation_files(EVALUATION_DIR)
if not evaluations:
    st.error(f"No evaluation files found in {EVALUATION_DIR}")
    st.stop()

study_list = sorted(evaluations.keys())

# Sidebar navigation
st.sidebar.header("Study Navigation")

if "study_index" not in st.session_state:
    st.session_state.study_index = 0

col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    if st.button("⬅ Previous"):
        st.session_state.study_index = max(0, st.session_state.study_index - 1)
        st.rerun()

with col_next:
    if st.button("Next ➡"):
        st.session_state.study_index = min(len(study_list) - 1, st.session_state.study_index + 1)
        st.rerun()

study = st.sidebar.selectbox(
    "Select Study",
    study_list,
    index=st.session_state.study_index,
)
st.session_state.study_index = study_list.index(study)

# Load data for selected study
eval_data = load_evaluation(evaluations[study])
study_images = get_study_images(study, IMAGES_ROOT)
factor_source_page = get_source_page(study, EXTRACTION_FACTORS_DIR)
correlation_source_page = get_source_page(study, EXTRACTION_CORRELATIONS_DIR)
metadata_source_page = get_metadata_source_page(study, Path("data/extraction_metadata") / selected_dataset)

st.header(f"Study: {study}")

# -------- Section 1: Metadata --------
st.subheader("📋 Metadata (Extracted vs. True)")

render_source_page_image(metadata_source_page, study_images, "Metadata")

metadata = eval_data.get("metadata", {})
if metadata:
    meta_df = format_comparison_table(metadata, "metadata")
    styled_df = meta_df.style.apply(highlight_match, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    total = len(meta_df)
    correct = len(meta_df[meta_df["Match"] == "✓"])
    accuracy = correct / total if total > 0 else 0
    st.metric("Metadata Accuracy", f"{accuracy:.1%}", f"{correct}/{total} correct")
else:
    st.warning("No metadata evaluation found")

st.divider()

# -------- Section 2: Factor Loadings --------
st.subheader("📊 Factor Loadings (Extracted vs. True)")

render_source_page_image(factor_source_page, study_images, "Factor loadings")

factor_loadings = eval_data.get("factor_loadings", {})
if factor_loadings:
    non_zero_factors = {
        k: v for k, v in factor_loadings.items()
        if isinstance(v, dict) and v.get("true") not in (None, 0)
    }

    if non_zero_factors:
        factor_df = format_comparison_table(non_zero_factors, "factors")
        styled_df = factor_df.style.apply(highlight_match, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        total = len(factor_df)
        correct = len(factor_df[factor_df["Match"] == "✓"])
        accuracy = correct / total if total > 0 else 0
        st.metric("Factor Loadings Accuracy (non-zero only)", f"{accuracy:.1%}", f"{correct}/{total} correct")
    else:
        st.info("All factor loadings are zero or null")

    if st.checkbox("Show all factor loadings (including zeros)"):
        all_factors_df = format_comparison_table(factor_loadings, "factors")
        st.dataframe(all_factors_df, use_container_width=True, hide_index=True, height=600)
else:
    st.warning("No factor loadings evaluation found")

st.divider()

# -------- Section 3: Factor Correlations --------
st.subheader("🔗 Factor Correlations (Extracted vs. True)")

render_source_page_image(correlation_source_page, study_images, "Correlations")

factor_correlations = eval_data.get("factor_correlations", {})
if factor_correlations:
    non_null_corrs = {
        k: v for k, v in factor_correlations.items()
        if isinstance(v, dict) and v.get("true") is not None
    }

    if non_null_corrs:
        corr_df = format_comparison_table(non_null_corrs, "correlations")
        styled_df = corr_df.style.apply(highlight_match, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        total = len(corr_df)
        correct = len(corr_df[corr_df["Match"] == "✓"])
        accuracy = correct / total if total > 0 else 0
        st.metric("Correlations Accuracy (reported values only)", f"{accuracy:.1%}", f"{correct}/{total} correct")
    else:
        st.info("All correlations are null (factors 4-5 not present or orthogonal rotation)")

    if st.checkbox("Show all correlations (including nulls)"):
        all_corrs_df = format_comparison_table(factor_correlations, "correlations")
        st.dataframe(all_corrs_df, use_container_width=True, hide_index=True)
else:
    st.warning("No factor correlations evaluation found")

st.divider()

# -------- Summary Statistics --------
st.subheader("📈 Overall Summary")

col1, col2, col3 = st.columns(3)

with col1:
    if factor_loadings:
        all_factor_correct = sum(1 for v in factor_loadings.values() if isinstance(v, dict) and v.get("accuracy") == 1)
        all_factor_total = len(factor_loadings)
        all_factor_acc = all_factor_correct / all_factor_total if all_factor_total > 0 else 0
        st.metric("All Factor Loadings", f"{all_factor_acc:.1%}", f"{all_factor_correct}/{all_factor_total}")

with col2:
    if factor_correlations:
        all_corr_correct = sum(1 for v in factor_correlations.values() if isinstance(v, dict) and v.get("accuracy") == 1)
        all_corr_total = len(factor_correlations)
        all_corr_acc = all_corr_correct / all_corr_total if all_corr_total > 0 else 0
        st.metric("All Correlations", f"{all_corr_acc:.1%}", f"{all_corr_correct}/{all_corr_total}")

with col3:
    if metadata:
        all_meta_correct = sum(1 for v in metadata.values() if isinstance(v, dict) and v.get("accuracy") == 1)
        all_meta_total = len(metadata)
        all_meta_acc = all_meta_correct / all_meta_total if all_meta_total > 0 else 0
        st.metric("All Metadata", f"{all_meta_acc:.1%}", f"{all_meta_correct}/{all_meta_total}")

# -------- Sidebar Info --------
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {selected_dataset}")
st.sidebar.caption(f"Evaluation dir: {EVALUATION_DIR}")
st.sidebar.caption(f"Total studies: {len(study_list)}")
st.sidebar.caption(f"Current: {st.session_state.study_index + 1}/{len(study_list)}")
