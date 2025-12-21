import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Summary Reviewer", layout="wide")

# -------- Settings --------
SUMMARIES_DIR = Path("summaries")
IMAGES_ROOT = Path("data/image_data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLUMNS = ["Item", "F1", "F2", "F3"]
DEFAULT_N_ROWS = 20


# -------- Helpers --------
def list_summaries(summaries_dir: Path):
    files = sorted(summaries_dir.glob("*.json"))
    return {path.stem: path for path in files}


def list_tuning_studies(images_root: Path) -> list[str]:
    tuning_dir = images_root / "tuning"
    if not tuning_dir.exists():
        return []
    return sorted([path.name for path in tuning_dir.iterdir() if path.is_dir()])


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_detection(study: str) -> dict:
    detection_path = Path("data/detection") / f"{study}.json"
    if not detection_path.exists():
        return {}
    return json.loads(detection_path.read_text(encoding="utf-8"))


def load_consistency(study: str) -> dict:
    consistency_path = Path("consistency_checks") / f"{study}.json"
    if not consistency_path.exists():
        return {}
    return json.loads(consistency_path.read_text(encoding="utf-8"))


def _filter_relevant_pages(pages: list, detection: dict) -> list:
    if not detection:
        return [page for page in pages if page.get("relevance")]
    flagged = set()
    for page in detection.get("pages", []):
        label = str(page.get("result", "")).strip().upper()
        if label in {"A", "C"}:
            flagged.add(page.get("number"))
    if not flagged:
        return [page for page in pages if page.get("relevance")]
    return [page for page in pages if page.get("number") in flagged]


def resolve_image_path(study: str, page_number: int, summary: dict):
    for entry in summary.get("relevant_pages", []):
        if entry.get("number") == page_number and entry.get("path"):
            return Path(entry["path"])
    matches = list(IMAGES_ROOT.glob(f"*/{study}/page{page_number}.png"))
    return matches[0] if matches else None


def load_table_for_name(save_name: str) -> tuple[pd.DataFrame, bool]:
    """Load existing CSV for this name if it exists, else create default empty table."""
    csv_path = OUTPUT_DIR / f"{save_name}.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path), True
        except Exception:
            pass

    df = pd.DataFrame({c: [""] * DEFAULT_N_ROWS for c in DEFAULT_COLUMNS})
    if "Item" in df.columns:
        for i in range(min(DEFAULT_N_ROWS, 20)):
            df.loc[i, "Item"] = str(i + 1)
    return df, False


def save_table(save_name: str, df: pd.DataFrame):
    csv_path = OUTPUT_DIR / f"{save_name}.csv"
    json_path = OUTPUT_DIR / f"{save_name}.json"

    df.to_csv(csv_path, index=False)

    records = df.to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def _build_table_from_extraction(extractions: list, page_number: int) -> pd.DataFrame:
    factor_loadings = None
    for extraction in extractions:
        pages = extraction.get("pages") or []
        if not pages:
            continue
        if pages[0].get("number") != page_number:
            continue
        raw_result = pages[0].get("result", "")
        payload = None
        if isinstance(raw_result, str):
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(1))
                except Exception:
                    payload = None
            if payload is None:
                match = re.search(r"```\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
                if match:
                    try:
                        payload = json.loads(match.group(1))
                    except Exception:
                        payload = None
            if payload is None:
                try:
                    payload = json.loads(raw_result)
                except Exception:
                    payload = None
        elif isinstance(raw_result, dict):
            payload = raw_result
        if not payload:
            continue
        samples = payload.get("samples") or []
        if not samples:
            continue
        factor_loadings = samples[0].get("factor_loadings") or {}
        break

    df = pd.DataFrame({c: [""] * DEFAULT_N_ROWS for c in DEFAULT_COLUMNS})
    if "Item" in df.columns:
        for i in range(min(DEFAULT_N_ROWS, 20)):
            df.loc[i, "Item"] = str(i + 1)

    if not factor_loadings:
        return df

    for i in range(1, DEFAULT_N_ROWS + 1):
        for factor in ("F1", "F2", "F3"):
            key = f"{factor}.{i}"
            if key in factor_loadings:
                df.loc[i - 1, factor] = factor_loadings.get(key)
    return df


def _confidence_color(label: str) -> str:
    mapping = {
        "extremely strong": "#1b5e20",
        "strong": "#2e7d32",
        "moderate": "#f9a825",
        "weak evidence": "#ef6c00",
        "very uncertain": "#c62828",
        "unknown": "#757575",
    }
    return mapping.get(label, "#757575")


def _render_confidence(label: str) -> None:
    color = _confidence_color(label)
    st.markdown(
        f"<span style='color:{color}; font-size:1.2em;'>●</span> {label}",
        unsafe_allow_html=True,
    )


def _render_status(label: str, ok: bool, invert: bool = False) -> None:
    is_ok = not ok if invert else ok
    color = "#2e7d32" if is_ok else "#c62828"
    status = "TRUE" if ok else "FALSE"
    st.markdown(
        f"<span style='color:{color}; font-size:1.2em;'>●</span> {label}: {status}",
        unsafe_allow_html=True,
    )


# -------- UI --------
st.title("Summary Reviewer")

summaries = list_summaries(SUMMARIES_DIR)
available_tuning = list_tuning_studies(IMAGES_ROOT)
study_list_source = available_tuning if available_tuning else sorted(summaries.keys())
if not study_list_source:
    st.error("No studies found in tuning images or summaries.")
    st.stop()
study_list_source = sorted(study_list_source)
summary_map = {study: summaries.get(study) for study in study_list_source}

st.sidebar.header("Study")
study_list = study_list_source
if "study_index" not in st.session_state:
    st.session_state.study_index = 0
if st.session_state.study_index >= len(study_list):
    st.session_state.study_index = 0

col_prev, col_next = st.sidebar.columns(2)
with col_prev:
    if st.button("Previous study"):
        st.session_state.study_index = max(0, st.session_state.study_index - 1)
        st.rerun()
with col_next:
    if st.button("Next study"):
        st.session_state.study_index = min(
            len(study_list) - 1, st.session_state.study_index + 1
        )
        st.rerun()

study = st.sidebar.selectbox(
    "Select study",
    study_list,
    index=st.session_state.study_index,
)
st.session_state.study_index = study_list.index(study)
summary_path = summary_map.get(study)
if not summary_path:
    st.warning(f"No summary found for {study}. Run summarize.py for this study.")
    st.stop()
summary = load_summary(summary_path)
pages = summary.get("pages", [])
if not pages:
    st.error(f"No pages found in summary: {summary_path}")
    st.stop()

consistency = load_consistency(study)
detection = load_detection(study)
pages = _filter_relevant_pages(pages, detection)
if not pages:
    st.warning("No relevant pages found for factor loading extraction.")
    st.stop()

st.sidebar.header("Navigation")
idx = st.sidebar.number_input(
    "Page index",
    min_value=1,
    max_value=len(pages),
    value=1,
    step=1,
) - 1
page = pages[idx]
page_number = page.get("number")
image_path = resolve_image_path(study, page_number, summary)
extractions = summary.get("extractions") or []

st.sidebar.header("Save As")
save_name = st.sidebar.text_input("Output name", value=study).strip() or study

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Page image")
    st.caption(f"Study: {study} | Page: {page_number}")
    if image_path and image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning("No image found for this page.")

with colB:
    if consistency:
        st.subheader("Consistency check")
        page_key = f"{study}{chr(ord('a') + idx)}" if len(consistency) > 1 else study
        page_check = consistency.get(page_key)
        if page_check:
            _render_status("All fields extracted", page_check.get("all_fields", False))
            _render_status(
                "Values within [-1, 1]",
                page_check.get("possible_values", False),
            )
            _render_status(
                "Table likely continued",
                page_check.get("table_likely_continued", False),
                invert=True,
            )
            _render_status(
                "All extracted values are zero",
                page_check.get("all_zeros", False),
                invert=True,
            )
        else:
            st.caption("No consistency record for this page.")
    st.markdown("Certainty about relevance of page:")
    _render_confidence(page.get("confidence", "unknown"))

    st.subheader("Extracted table (editable)")
    extracted_df = _build_table_from_extraction(extractions, page_number)
    if image_path and image_path.exists():
        df, has_saved = load_table_for_name(save_name)
        if not has_saved:
            df = extracted_df
    else:
        df = extracted_df

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{study}_{page_number}",
    )

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("Submit / Save", type="primary"):
            csv_path, json_path = save_table(save_name, edited)
            st.success(f"Saved: {csv_path.name} and {json_path.name}")

    with c2:
        if st.button("Reset to empty"):
            for ext in ("csv", "json"):
                p = OUTPUT_DIR / f"{save_name}.{ext}"
                if p.exists():
                    p.unlink()
            st.rerun()

    with c3:
        st.caption(f"Edits are saved to: {OUTPUT_DIR.resolve()}")

st.sidebar.write("---")
st.sidebar.caption("Run: streamlit run app.py")
