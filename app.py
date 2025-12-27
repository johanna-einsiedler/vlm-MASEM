#!/usr/bin/env python3
"""Streamlit app for reviewing extraction evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Extraction Evaluation Viewer", layout="wide")

# -------- Settings --------
EVALUATION_DIR = Path("data/evaluation/tuning")
IMAGES_ROOT = Path("data/image_data/tuning")
DETECTION_DIR = Path("data/detection")


# -------- Helper Functions --------
def list_evaluation_files() -> Dict[str, Path]:
    """List all evaluation JSON files."""
    if not EVALUATION_DIR.exists():
        return {}
    files = sorted(EVALUATION_DIR.glob("*.json"))
    return {path.stem: path for path in files}


def load_evaluation(path: Path) -> dict:
    """Load evaluation JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_detection(study_base: str) -> dict:
    """Load detection results for a study."""
    # Remove suffix (a, b, c) to get base study name
    study = study_base.rstrip("abc")
    detection_path = DETECTION_DIR / f"{study}.json"
    if not detection_path.exists():
        return {}
    return json.loads(detection_path.read_text(encoding="utf-8"))


def get_study_images(study_base: str) -> Dict[int, Path]:
    """Get all page images for a study."""
    study = study_base.rstrip("abc")
    study_dir = IMAGES_ROOT / study
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


def get_pages_by_label(detection: dict, labels: tuple) -> List[int]:
    """Get page numbers with specific detection labels."""
    pages = []
    for page in detection.get("pages", []):
        label = str(page.get("result", "")).strip().upper()
        if label in labels:
            pages.append(page.get("number"))
    return sorted(pages)


def format_comparison_table(data: dict, data_type: str) -> pd.DataFrame:
    """Format extracted vs true values as a DataFrame.

    Args:
        data: Dictionary with keys as field names and values as dicts with 'extracted', 'true', 'accuracy'
        data_type: Type of data ('factors', 'correlations', 'metadata')
    """
    rows = []
    for key, values in data.items():
        if not isinstance(values, dict):
            continue

        extracted = values.get("extracted")
        true = values.get("true")
        accuracy = values.get("accuracy", 0)

        # Format values
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


def render_value_comparison(label: str, extracted, true, accuracy: int):
    """Render a single value comparison."""
    match_color = "#2e7d32" if accuracy == 1 else "#c62828"
    match_symbol = "✓" if accuracy == 1 else "✗"

    st.markdown(
        f"**{label}**: {extracted} → {true} "
        f"<span style='color:{match_color}; font-size:1.2em;'>{match_symbol}</span>",
        unsafe_allow_html=True
    )


# -------- Main App --------
st.title("Extraction Evaluation Viewer")

# Load all evaluation files
evaluations = list_evaluation_files()
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

# Load evaluation data
eval_data = load_evaluation(evaluations[study])
detection = load_detection(study)
study_images = get_study_images(study)

# Get relevant pages
factor_pages = get_pages_by_label(detection, ("A", "C"))
correlation_pages = get_pages_by_label(detection, ("B", "C"))

st.header(f"Study: {study}")

# -------- Section 1: Metadata --------
st.subheader("📋 Metadata (Extracted vs. True)")

metadata = eval_data.get("metadata", {})
if metadata:
    meta_df = format_comparison_table(metadata, "metadata")

    # Color code the dataframe
    def highlight_match(row):
        if row["Match"] == "✓":
            return ['background-color: #e8f5e9'] * len(row)
        else:
            return ['background-color: #ffebee'] * len(row)

    styled_df = meta_df.style.apply(highlight_match, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Summary stats
    total = len(meta_df)
    correct = len(meta_df[meta_df["Match"] == "✓"])
    accuracy = correct / total if total > 0 else 0
    st.metric("Metadata Accuracy", f"{accuracy:.1%}", f"{correct}/{total} correct")
else:
    st.warning("No metadata evaluation found")

st.divider()

# -------- Section 2: Factor Loadings --------
st.subheader("📊 Factor Loadings (Extracted vs. True)")

# Show factor loading images
if factor_pages:
    st.caption(f"Factor loading pages: {factor_pages}")

    # Display images
    img_cols = st.columns(min(len(factor_pages), 3))
    for i, page_num in enumerate(factor_pages[:3]):  # Show max 3 images
        if page_num in study_images:
            with img_cols[i % 3]:
                img = Image.open(study_images[page_num])
                st.image(img, caption=f"Page {page_num}", use_container_width=True)

factor_loadings = eval_data.get("factor_loadings", {})
if factor_loadings:
    # Show only non-zero true values for clarity
    non_zero_factors = {
        k: v for k, v in factor_loadings.items()
        if isinstance(v, dict) and v.get("true") not in (None, 0)
    }

    if non_zero_factors:
        factor_df = format_comparison_table(non_zero_factors, "factors")

        def highlight_match(row):
            if row["Match"] == "✓":
                return ['background-color: #e8f5e9'] * len(row)
            else:
                return ['background-color: #ffebee'] * len(row)

        styled_df = factor_df.style.apply(highlight_match, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

        # Summary stats
        total = len(factor_df)
        correct = len(factor_df[factor_df["Match"] == "✓"])
        accuracy = correct / total if total > 0 else 0
        st.metric("Factor Loadings Accuracy (non-zero only)", f"{accuracy:.1%}", f"{correct}/{total} correct")
    else:
        st.info("All factor loadings are zero or null")

    # Show option to view all values
    if st.checkbox("Show all factor loadings (including zeros)"):
        all_factors_df = format_comparison_table(factor_loadings, "factors")
        st.dataframe(all_factors_df, use_container_width=True, hide_index=True, height=600)
else:
    st.warning("No factor loadings evaluation found")

st.divider()

# -------- Section 3: Factor Correlations --------
st.subheader("🔗 Factor Correlations (Extracted vs. True)")

# Show correlation images
if correlation_pages:
    st.caption(f"Correlation pages: {correlation_pages}")

    # Display images
    img_cols = st.columns(min(len(correlation_pages), 3))
    for i, page_num in enumerate(correlation_pages[:3]):  # Show max 3 images
        if page_num in study_images:
            with img_cols[i % 3]:
                img = Image.open(study_images[page_num])
                st.image(img, caption=f"Page {page_num}", use_container_width=True)

factor_correlations = eval_data.get("factor_correlations", {})
if factor_correlations:
    # Show only non-null true values
    non_null_corrs = {
        k: v for k, v in factor_correlations.items()
        if isinstance(v, dict) and v.get("true") is not None
    }

    if non_null_corrs:
        corr_df = format_comparison_table(non_null_corrs, "correlations")

        def highlight_match(row):
            if row["Match"] == "✓":
                return ['background-color: #e8f5e9'] * len(row)
            else:
                return ['background-color: #ffebee'] * len(row)

        styled_df = corr_df.style.apply(highlight_match, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Summary stats
        total = len(corr_df)
        correct = len(corr_df[corr_df["Match"] == "✓"])
        accuracy = correct / total if total > 0 else 0
        st.metric("Correlations Accuracy (reported values only)", f"{accuracy:.1%}", f"{correct}/{total} correct")
    else:
        st.info("All correlations are null (factors 4-5 not present or orthogonal rotation)")

    # Show option to view all values
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
st.sidebar.caption(f"Evaluation files: {EVALUATION_DIR}")
st.sidebar.caption(f"Total studies: {len(study_list)}")
st.sidebar.caption(f"Current: {st.session_state.study_index + 1}/{len(study_list)}")
