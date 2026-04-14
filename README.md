# VLM Extraction Pipeline for MASEM

This pipeline uses a vision language model (VLM) to extract structured MASEM data — item-level factor loadings, factor correlations, and study-level metadata — directly from published PDF articles. Extracted values are stored as structured JSON and reviewed by a human coder using [PaperLens](https://paperlens.fly.dev/) before entering the final meta-analytic dataset.

---

## Quickstart (default: GPT-5 mini)

```bash
# Extract all papers in the tuning set
python pipeline_steps/run.py --dataset tuning

# Single paper
python pipeline_steps/run.py --dataset tuning --pdf wise2000.pdf

# Force re-run even if outputs exist
python pipeline_steps/run.py --dataset tuning --force

# Evaluate + figures only (skip extraction)
python pipeline_steps/evaluate.py --dataset tuning_openai --plots
```

Papers longer than 50 pages are skipped with a warning — provide a page-trimmed version.

---

## Stage 1 – Clarifying the Roles of the AI in the Research Cycle

### 1. Where in the research workflow is AI used?

AI is used as a **labelling-assistance tool** at the data extraction stage of the meta-analytic workflow. Each PDF article is rendered into page images and passed to a VLM with a structured prompt. The model returns JSON with extracted numerical values and verbatim evidence snippets. A human coder then reviews, corrects, and approves each extraction in [PaperLens](https://paperlens.fly.dev/) before the values enter the pooled dataset. AI replaces the first pass of manual coding; human verification replaces the second pass.

### 2. What are the inputs and outputs?

**Inputs:**
- PDF articles: `data/intermediate_data/<dataset>/*.pdf`
- Three plain-text extraction prompts (one per task): `prompts/`

Each PDF is rendered to per-page PNG images at 200 DPI via PyMuPDF. Images are sent in chunks (≤ 10–15 pages per call) alongside the prompt. Three separate calls are made per paper: factor loadings, factor correlations, and study metadata.

**Outputs** — three JSON files per study, written to:
```
data/extraction_factors/<dataset>_<provider>/<study>.json
data/extraction_correlations/<dataset>_<provider>/<study>.json
data/extraction_metadata/<dataset>_<provider>/<study>.json
```

Each file uses an envelope format:
```json
{
  "filename": "wise2000.pdf",
  "pages_processed": 8,
  "model": "gpt-5-mini",
  "prompt": "...",
  "entries": [
    {
      "sample_id": "sample1",
      "factor_loadings": {"F1.1": 0.64, "F1.2": 0.54, "...": "..."},
      "evidence": [
        {"snippet": "Table 3 ...", "page": 7, "source": "Table 3", "field": "F1 loadings"}
      ],
      "notes": ""
    }
  ],
  "human_overrides": [],
  "original_model_response": "..."
}
```

The `evidence` array provides verbatim quotes and page numbers for each extracted value, enabling human reviewers to verify extractions in [PaperLens](https://paperlens.fly.dev/).

### 3. Would different prompts, models, or versions meaningfully change the output?

Yes — model choice, prompt wording, and model version all affect extraction quality. 

A separate tuning set (`data/intermediate_data/tuning/`, n=10) with ground truth codings is used to develop and benchmark prompts before applying them to the full evaluation set.

We record the model version in each output file (`"model": "gpt-5-mini"`), so the exact version used for any extraction is always traceable.

---

## Stage 2 – Determining What Needs to Be Reproducible

### 1. Are all relevant inputs — prompts, settings, model details — documented?

Yes. All inputs required to reproduce or audit an extraction are either versioned in the repository or recorded in the output files:

**Prompts** 
```
prompts/prompt_factor_loadings.txt
prompts/prompt_correlations.txt
prompts/prompt_metadata.txt
```

**Model and provider** — defaults defined in `pipeline_steps/extract.py` and recorded per-file in output JSON:
```python
DEFAULT_MODEL = {
    "openai":   "gpt-5-mini",
    "cohere":   "command-a-vision-07-2025",
    "gemini":   "gemini-2.0-flash",
    "together": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
}
```

**Inference settings** — fixed defaults, all overridable via CLI:

| Setting | Default | CLI flag |
|---|---|---|
| Temperature | 0.1 | *(hardcoded; 0 for reasoning models)* |
| Max output tokens | 16,000 | `--max-output-tokens` |
| PDF render resolution | 200 DPI | `--dpi` |
| Chunk size | 15 pages (10 for Together) | `--chunk-size` |

**Input data:**
```
data/intermediate_data/tuning/     ← prompt development PDFs (n=10)
data/intermediate_data/eval/       ← main evaluation set
data/ground_truth_codings.xlsx     ← ground truth for accuracy benchmarking
```

### 2. Does reproducibility matter for the AI-assisted process, its outputs, or both?


In this pipeline, AI-extracted values are treated as provisional and verified against the original text passage or table before inclusion in the final analytic dataset via [PaperLens](https://paperlens.fly.dev/). While extraction variability without human verification could affect the pooled correlation matrix and downstream MASEM results, the actual need for strict output reproducibility is reduced by the human-in-the-loop setup: if different LLM runs produce different extracted values, the probability that a human reviewer catches and corrects the error is high.

The `human_overrides` field in each output JSON records every correction made during review, so the final verified dataset is fully auditable regardless of model variability.

---

## Stage 3 – Deciding How to Document AI Assistance

### 1. Is exact reproduction of the AI output technically possible and feasible?

Exact reproduction is technically not feasible in this setting. An external API provider (OpenAI, `gpt-5-mini`) was used for extraction; changes in model configuration by the provider and inherent stochasticity in deployment prevent exact reproduction. The model name is recorded in every output file (`"model": "gpt-5-mini"`), and the raw model response is archived in `original_model_response`, but future API calls may silently resolve to a different model checkpoint.

### 2. Which technical details must be described in the manuscript?

The manuscript should describe the instantiated workflow: full-text PDFs served as source material, processed through a structured extraction pipeline comprising prompt preparation from a coding scheme, document parsing and chunking, LLM-based extraction, and output logging. It should identify the specific components: the LLM provider and model (`openai`, `gpt-5-mini`) and the prompt variant used during inference. It should explain that prompts were derived from a structured coding scheme via a formal prompt-construction workflow, with exact details documented in the repository. Finally, it should make clear that the analysis was conducted through a command-line pipeline rather than an informal chat interface, as this is important for reproducibility and for understanding which aspects of the workflow were standardised.

### 3. Which technical details must be documented in the repository?

| What | Where in this repo |
|---|---|
| Python version, installation, dependencies | `README.md` (this file), `requirements.txt` |
| CLI commands used for the analysis | `README.md` quickstart section |
| Prompt files and coding scheme | `prompts/` |
| Model, provider, and inference settings | `pipeline_steps/extract.py` (`DEFAULT_MODEL`, defaults) |
| Document parsing and chunking logic | `pipeline_steps/extract.py` (`_pdf_to_base64_images`, `_chunk_images`) |
| Raw LLM outputs per study | `original_model_response` field in each extraction JSON |
| Human-verified values and corrections | `entries` + `human_overrides` fields in each extraction JSON |
| Ground truth codings | `data/ground_truth_codings.xlsx` |
| Accuracy evaluation results | `data/evaluation/<dataset>_<provider>/` |

### 4. Does the documentation adhere to FAIR principles?

For FAIR compliance, the repository contains not only the code but also a concise run protocol (this README) specifying the exact sequence of commands used in the study — from extraction through evaluation to figures — so that another researcher can reproduce the implemented pipeline.

| Principle | How addressed |
|---|---|
| **Findable** | Repository on OSF/GitHub; model name and prompt version recorded in every output file |
| **Accessible** | All prompts, code, and extracted data are plain text / JSON; no proprietary formats |
| **Interoperable** | Output schema is consistent across providers; ground truth in standard xlsx |
| **Reusable** | Pipeline is parameterised (provider, model, dataset) and can be applied to other scales or corpora with new prompts |

---

## Data layout

```
data/intermediate_data/
    tuning/                                      ← prompt development set (n=10)
    eval/                                        ← main evaluation set

data/extraction_factors/<dataset>_<provider>/<study>.json
data/extraction_correlations/<dataset>_<provider>/<study>.json
data/extraction_metadata/<dataset>_<provider>/<study>.json

data/evaluation/<dataset>_<provider>/<study>.json
data/ground_truth_codings.xlsx

figures/accuracy_plot_factors_<dataset>_<provider>.png
figures/accuracy_plot_correlations_<dataset>_<provider>.png
figures/accuracy_plot_metadata_<dataset>_<provider>.png
figures/accuracy_plot_metadata_fields_<dataset>_<provider>.png
```

---

## Pipeline scripts

| Script | Role |
|---|---|
| `pipeline_steps/extract.py` | PDF → page images → VLM → JSON (3 tasks per paper) |
| `pipeline_steps/evaluate.py` | Compare extracted values to ground truth; `--plots` generates figures |
| `pipeline_steps/accuracy_plots.py` | Generate accuracy figures per dataset |
| `pipeline_steps/run.py` | Orchestrate extract → evaluate → plots for a full dataset |
| `pipeline_steps/pdf_to_images.py` | Pre-render page images |

---

## Dataset split

- **Tuning** (`tuning/`, n=10): prompt development and accuracy benchmarking against ground truth
- **Evaluation** (`eval/`): remaining papers for the main meta-analysis
