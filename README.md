# VLM Pipeline

This repository runs a PDF-to-VLM pipeline for TAS-20 factor analysis table detection, extraction, and evaluation.

## High-level flow

**Per-study pipeline (Steps 1-7):**
1. Convert PDFs to page images
2. Detect pages with factor/correlation tables (A/B/C/D labels)
3. Refine table regions using PaddleOCR
4. Parse pages to markdown using VLM (Qwen2.5-VL 72B)
5. Extract factor loadings and correlations using LLM (DeepSeek GPT OSS 120B)
6. Extract metadata using LLM
7. Evaluate extracted values against ground truth

**Final step (per dataset):**
8. Generate accuracy plots (4 plots: factors, correlations, metadata, metadata fields)

**Review:**
- Use Streamlit app to review extraction results with color-coded comparison tables

## Key scripts

- `pipeline.py`: End-to-end orchestration (single PDF or dataset)
- `pipeline_steps/00_pdf_to_images.py`: Render PDFs to `data/image_data/<dataset>/<study>/pageN.png`
- `pipeline_steps/01_detection.py`: Page-level detection using Qwen VLM (A/B/C/D labels)
- `pipeline_steps/02_table_refinement.py`: Refine table regions using PaddleOCR TableRecognitionPipelineV2
- `pipeline_steps/03_parse_pdf.py`: Convert pages to markdown using Qwen2.5-VL 72B
- `pipeline_steps/04_factor_extraction.py`: Extract factor loadings and correlations using DeepSeek LLM
  - **Dual extraction task**: Runs two separate prompts (factors on A/C pages, correlations on B/C pages)
  - **Auto-generates** correlation files with 0 values when no B/C pages exist (orthogonal rotation)
- `pipeline_steps/05_metadata_extraction.py`: Extract study metadata using DeepSeek LLM
- `pipeline_steps/06_evaluate.py`: Compare extraction results to `data/ground_truth_codings.xlsx`
  - Handles multi-sample studies using permutation matching
  - Skips re-evaluation if all three sections (factors, correlations, metadata) exist
- `pipeline_steps/07_accuracy_plots.py`: Generate 4 accuracy plots per dataset
- `app.py`: Streamlit evaluation viewer showing extracted vs. true values with source images

## Common commands

**Run full pipeline:**
```bash
python pipeline.py --pdf wise2000.pdf
python pipeline.py --dataset tuning
python pipeline.py --dataset tuning --force  # Re-run all steps
```

**Run individual steps:**
```bash
# Step 1: Convert PDFs
python pipeline_steps/00_pdf_to_images.py tuning --dpi 300

# Step 2: Detect pages
python pipeline_steps/01_detection.py tuning

# Step 4: Parse to markdown
python pipeline_steps/03_parse_pdf.py tuning

# Step 5: Extract factors and correlations
python pipeline_steps/04_factor_extraction.py tuning

# Step 6: Extract metadata
python pipeline_steps/05_metadata_extraction.py tuning

# Step 7: Evaluate
python pipeline_steps/06_evaluate.py tuning

# Step 8: Generate plots
python pipeline_steps/07_accuracy_plots.py tuning
```

**Review results:**
```bash
streamlit run app.py
```

## Data layout

**Input:**
- `data/intermediate_data/<dataset>/*.pdf`: Input PDFs (`tuning`, `validation`, `eval`)

**Processing:**
- `data/image_data/<dataset>/<study>/pageN.png`: Rendered page images
- `data/detection/<study>.json`: Page detection results (A/B/C/D labels)
- `data/refined_tables/<dataset>/<study>/pageN.json`: PaddleOCR table refinement results
- `data/md_extraction/<dataset>/<study>.json`: Markdown conversion results

**Extraction outputs (3 separate files per study):**
- `data/extraction_factors/<dataset>/<study>.json`: Factor loadings (from A/C pages)
- `data/extraction_correlations/<dataset>/<study>.json`: Factor correlations (from B/C pages)
  - Auto-generated with 0 values if no B/C pages detected
- `data/extraction_metadata/<dataset>/<study>.json`: Study metadata

**Evaluation:**
- `data/evaluation/<dataset>/<study>[a|b|c].json`: Evaluation results (one file per sample)
  - Contains three sections: `factor_loadings`, `factor_correlations`, `metadata`

**Visualization:**
- `figures/accuracy_plot_factors_<dataset>.png`: Factor loadings accuracy per study
- `figures/accuracy_plot_correlations_<dataset>.png`: Correlations accuracy per study
- `figures/accuracy_plot_metadata_<dataset>.png`: Metadata accuracy per study
- `figures/accuracy_plot_metadata_fields_<dataset>.png`: Mean accuracy per metadata field

**Legacy:**
- `summaries/<study>.json`: Summary payloads (legacy)
- `consistency_checks/<study>.json`: Consistency checks (legacy)

## Prompts

**Detection:**
- `prompts/01_detection_prompt.txt`: Page classification (A/B/C/D labels)

**Extraction:**
- `prompts/02_extraction_prompt_factors.txt`: Factor loadings extraction
- `prompts/03_extraction_prompt_correlations.txt`: Factor correlations extraction
  - Includes special case for orthogonal rotation (all correlations = 0)
- `prompts/04_extraction_prompt_metadata.txt`: Study metadata extraction

## Multi-sample handling

Studies with multiple samples (e.g., parker1993 with german, american, canadian samples):
- Factor and correlation files use `{"samples": [...]}` format with `sample_id` field
- Metadata files use `{"records": [...]}` format with `source_sample_id` field
- Evaluation uses permutation matching to align extracted samples with ground truth entries
- Each sample gets a separate evaluation file (e.g., `parker1993a.json`, `parker1993b.json`)

## Technologies

- **VLM**: Qwen2.5-VL 72B (image-to-markdown conversion)
- **LLM**: DeepSeek GPT OSS 120B via Together API (structured extraction)
- **OCR**: PaddleOCR TableRecognitionPipelineV2 (table cell detection)
- **Visualization**: Streamlit + Matplotlib
- **Ground Truth**: Excel-based with tolerance-based numeric comparison
