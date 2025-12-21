# VLM Pipeline

This repository runs a PDF-to-VLM pipeline for TAS-20 table detection, extraction, and evaluation.

## High-level flow
1) Convert PDFs to page images.
2) Detect pages with factor/correlation tables (A/B/C/D labels).
3) Extract tables from relevant pages.
4) Evaluate extracted values against ground truth.
5) Generate summaries and consistency checks.
6) Review results in a Streamlit app.

## Key scripts
- `pipeline.py`: end-to-end run (single PDF or dataset).
- `pipeline_steps/convert_to_image`: render PDFs to `data/image_data/<dataset>/<study>/pageN.png`.
- `pipeline_steps/detection.py`: page-level detection using Qwen and detection prompt.
- `pipeline_steps/extraction.py`: page-level extraction using Qwen and extraction prompt.
- `pipeline_steps/evaluate.py`: compare extraction results to `data/ground_truth_codings.xlsx`.
- `pipeline_steps/consistency_check.py`: sanity checks on extraction outputs.
- `pipeline_steps/summarize.py`: builds `summaries/<study>.json` for the viewer.
- `pipeline_steps/accuracy_evaluation.py`: plots accuracy bars to `figures/accuracy_plot.png`.
- `app.py`: Streamlit viewer for summaries, images, and editable tables.

## Common commands
- Convert PDFs:
  - `python pipeline_steps/convert_to_image tune --dpi 300`
- Run the full pipeline:
  - `python pipeline.py --pdf wise2000.pdf`
  - `python pipeline.py --dataset tune`
- Summarize and view:
  - `python pipeline_steps/summarize.py wise2000`
  - `streamlit run app.py`
- Plot accuracy:
  - `python pipeline_steps/accuracy_evaluation.py --input data/evaluation`

## Data layout
- `data/intermediate_data/<dataset>/*.pdf`: input PDFs (`tuning`, `validation`, `eval`)
- `data/image_data/<dataset>/<study>/pageN.png`: rendered pages
- `data/detection/<study>.json`: detection outputs
- `data/extraction/<study>[a|b|c].json`: extraction outputs
- `data/evaluation/<dataset>/<study>[a|b|c].json`: evaluation outputs
- `summaries/<study>.json`: summary payloads for the viewer
- `consistency_checks/<study>.json`: consistency checks
- `figures/accuracy_plot.png`: accuracy plot

## Prompts
- `prompts/01_detection_prompt.txt`
- `prompts/02_extraction_prompt.txt`
- `prompts/04_robustness_factors.txt`
