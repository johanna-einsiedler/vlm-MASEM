#!/usr/bin/env python3
"""Single-step MASEM extraction using GPT with page images.

For each extraction task (factor loadings, correlations, metadata) the script
sends PDF pages as base64 images directly to GPT — no intermediate markdown.

Papers longer than `chunk_size` pages are split into chunks; each chunk gets
its own GPT call and the results are merged automatically.

Outputs mirror the regular pipeline format so downstream evaluation (steps 6-7)
works unchanged:
  data/extraction_factors/<dataset>_1step/<study>.json
  data/extraction_correlations/<dataset>_1step/<study>.json
  data/extraction_metadata/<dataset>_1step/<study>.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

try:
    import cohere as cohere_sdk
except ImportError:
    cohere_sdk = None  # Cohere is optional; only required if --provider cohere is used

try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
except ImportError:
    genai_sdk = None   # google-genai is optional; only required if --provider gemini is used
    genai_types = None

try:
    from together import Together as together_sdk
except ImportError:
    together_sdk = None  # together is optional; only required if --provider together is used

DEFAULT_MODEL = {
    "openai": "gpt-5-mini",
    "cohere": "command-a-vision-07-2025",
    "gemini": "gemini-2.0-flash",
    "together": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
}

# Maximum images allowed per API call for each provider.
MAX_IMAGES_PER_CALL = {
    "openai": 15,
    "cohere": 15,
    "gemini": 15,
    "together": 10,
}

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt_text(prompt_filename: str) -> str:
    """Load a prompt text file from the prompts/ directory."""
    candidates = [
        ROOT / "prompts" / prompt_filename,
        Path("prompts") / prompt_filename,
    ]
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        f"Prompt file '{prompt_filename}' not found. Looked in: {candidates}"
    )


def _pdf_to_base64_images(pdf_path: str | Path, dpi: int = 200) -> list[str]:
    """Convert every page of a PDF to a base64-encoded PNG string."""
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    doc.close()
    return images


def _build_vision_input(prompt_text: str, b64_images: list[str]) -> list[dict]:
    """Build a Responses-API input list: text prompt followed by page images."""
    content: list[dict] = [{"type": "input_text", "text": prompt_text}]
    for b64 in b64_images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
            "detail": "high",
        })
    return [{"role": "user", "content": content}]


def _parse_json_blob(text: str) -> dict:
    """Extract and parse the first JSON object from a text string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in text")
    return json.loads(match.group())


def _chunk_images(images: list[str], chunk_size: int = 15) -> list[list[str]]:
    """Split a flat list of base64 images into sub-lists of at most chunk_size."""
    return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]


def _merge_samples(results: list[str], offsets: list[int], data_key: str) -> str:
    """Merge sample results across chunks into a single JSON string.

    For each sample index:
    - data_key values (factor_loadings / factor_correlations): first non-null value
      per key wins across chunks.
    - evidence: all entries are kept; page numbers are offset-adjusted so they
      reflect absolute PDF page numbers (e.g. a finding on page 2 of chunk 3,
      where chunks 1+2 covered 10 pages, becomes page 12).
    - All other fields (sample_id, notes, …): taken from the first chunk that
      provides them.
    """
    parsed: list[list[dict]] = []
    for result, offset in zip(results, offsets):
        adjusted = _adjust_evidence_pages(result, offset)
        try:
            blob = _parse_json_blob(adjusted)
            samples = blob.get("samples", [])
            if samples:
                parsed.append(samples)
        except Exception:
            continue

    if not parsed:
        return results[0] if results else json.dumps({"samples": []})
    if len(parsed) == 1:
        return json.dumps({"samples": parsed[0]}, indent=2)

    n = len(parsed[0])
    merged_samples = []
    for i in range(n):
        candidates = [chunk[i] for chunk in parsed if i < len(chunk)]

        # Merge data values: first non-null value per key wins
        merged_data: dict = {}
        for candidate in candidates:
            for k, v in candidate.get(data_key, {}).items():
                if k not in merged_data or (merged_data[k] is None and v is not None):
                    merged_data[k] = v

        # Combine evidence from all chunks (deduplicate by snippet)
        all_evidence: list[dict] = []
        seen: set[str] = set()
        for candidate in candidates:
            for ev in candidate.get("evidence", []):
                key = ev.get("snippet") or str(ev)
                if key not in seen:
                    all_evidence.append(ev)
                    seen.add(key)

        # Base sample from first candidate; overwrite merged fields
        sample = dict(candidates[0])
        sample[data_key] = merged_data
        sample["evidence"] = all_evidence
        merged_samples.append(sample)

    return json.dumps({"samples": merged_samples}, indent=2)


def _merge_metadata(results: list[str], offsets: list[int]) -> str:
    """Merge metadata records across chunks: first non-null value per field wins.

    Each chunk returns {"records": [{...}]}. For each record index and each field,
    the first non-null value across chunks is kept. Evidence page numbers are
    offset-adjusted to absolute PDF page numbers before merging.
    """
    parsed: list[list[dict]] = []
    for r, offset in zip(results, offsets):
        adjusted = _adjust_evidence_pages(r, offset)
        try:
            blob = _parse_json_blob(adjusted)
            recs = blob.get("records", [])
            if recs:
                parsed.append(recs)
        except Exception:
            continue

    if not parsed:
        return results[0] if results else json.dumps({"records": []})
    if len(parsed) == 1:
        return json.dumps({"records": parsed[0]}, indent=2)

    n = len(parsed[0])
    merged = []
    for i in range(n):
        candidates = [chunk[i] for chunk in parsed if i < len(chunk)]
        all_fields: list[str] = list(dict.fromkeys(k for rec in candidates for k in rec))
        record: dict = {}
        for field in all_fields:
            for rec in candidates:
                val = rec.get(field)
                if val is not None:
                    record[field] = val
                    break
            else:
                record[field] = None
        merged.append(record)

    return json.dumps({"records": merged}, indent=2)


def _call_gpt(
    client: OpenAI,
    system_prompt: str,
    input_content: list[dict],
    model: str,
    max_output_tokens: int,
    step_label: str,
) -> str:
    """Call the OpenAI Responses API and return the raw text output."""
    is_reasoning = any(
        model.startswith(p) for p in ("o3", "o4", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5")
    )
    kwargs: dict = dict(
        model=model,
        instructions=system_prompt,
        input=input_content,
        max_output_tokens=max_output_tokens,
    )
    if not is_reasoning:
        kwargs["temperature"] = 0.1

    response = client.responses.create(**kwargs)

    usage = response.usage
    print(
        f"  [{step_label}] model={response.model} "
        f"in={usage.input_tokens:,} out={usage.output_tokens:,} "
        f"total={usage.total_tokens:,}"
    )

    for item in response.output:
        if hasattr(item, "content") and item.content is not None:
            for block in item.content:
                if hasattr(block, "text"):
                    return block.text
        if hasattr(item, "text") and item.text is not None:
            return item.text
    raise ValueError(f"[{step_label}] No text found in response output")


def _call_cohere(
    client,
    system_prompt: str,
    prompt_text: str,
    b64_images: list[str],
    model: str,
    max_output_tokens: int,
    step_label: str,
) -> str:
    """Call Cohere's chat API with vision and return the raw text output."""
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    response = client.chat(
        model=model,
        messages=messages,
        max_tokens=max_output_tokens,
    )

    # Usage reporting (Cohere v2 structure)
    try:
        u = response.usage
        in_tok = getattr(getattr(u, "tokens", None), "input_tokens", None) \
               or getattr(getattr(u, "billed_units", None), "input_tokens", "?")
        out_tok = getattr(getattr(u, "tokens", None), "output_tokens", None) \
                or getattr(getattr(u, "billed_units", None), "output_tokens", "?")
        print(f"  [{step_label}] model={model} in={in_tok} out={out_tok}")
    except Exception:
        print(f"  [{step_label}] model={model}")

    return response.message.content[0].text


def _call_gemini(
    client,
    system_prompt: str,
    prompt_text: str,
    b64_images: list[str],
    model: str,
    max_output_tokens: int,
    step_label: str,
) -> str:
    """Call the Gemini API (google-genai SDK) with vision and return the raw text output.

    Args:
        client: A google.genai.Client instance.
    """
    parts = [genai_types.Part.from_text(text=prompt_text)]
    for b64 in b64_images:
        parts.append(
            genai_types.Part.from_bytes(
                data=base64.b64decode(b64),
                mime_type="image/png",
            )
        )

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
        temperature=0.1,
    )

    response = client.models.generate_content(
        model=model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=config,
    )

    try:
        u = response.usage_metadata
        print(
            f"  [{step_label}] model={model} "
            f"in={u.prompt_token_count} out={u.candidates_token_count}"
        )
    except Exception:
        print(f"  [{step_label}] model={model}")

    return response.text


def _call_together(
    client,
    system_prompt: str,
    prompt_text: str,
    b64_images: list[str],
    model: str,
    max_output_tokens: int,
    step_label: str,
    timeout: int = 180,
    max_retries: int = 3,
) -> str:
    """Call Together's chat completions API with vision and return the raw text output.

    Retries up to max_retries times on timeout or transient errors, with
    exponential backoff. Raises RuntimeError if all attempts fail.
    """
    import concurrent.futures
    import time

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    def _attempt() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=0.1,
        )
        try:
            u = response.usage
            print(
                f"  [{step_label}] model={model} "
                f"in={u.prompt_tokens} out={u.completion_tokens} "
                f"total={u.total_tokens}"
            )
        except Exception:
            print(f"  [{step_label}] model={model}")
        return response.choices[0].message.content

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_attempt)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            last_exc = TimeoutError(
                f"[{step_label}] Together API call timed out after {timeout}s "
                f"(attempt {attempt}/{max_retries})"
            )
            print(f"  WARNING: {last_exc}")
        except Exception as exc:
            last_exc = exc
            print(f"  WARNING: [{step_label}] attempt {attempt}/{max_retries} failed: {exc}")

        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"[{step_label}] All {max_retries} attempts failed. Last error: {last_exc}"
    )


def _call_model(
    client,
    provider: str,
    system_prompt: str,
    prompt_text: str,
    b64_images: list[str],
    model: str,
    max_output_tokens: int,
    step_label: str,
) -> str:
    """Dispatch to the correct provider's API call."""
    if provider == "openai":
        input_content = _build_vision_input(prompt_text, b64_images)
        return _call_gpt(client, system_prompt, input_content, model, max_output_tokens, step_label)
    elif provider == "cohere":
        return _call_cohere(client, system_prompt, prompt_text, b64_images, model, max_output_tokens, step_label)
    elif provider == "gemini":
        return _call_gemini(client, system_prompt, prompt_text, b64_images, model, max_output_tokens, step_label)
    elif provider == "together":
        return _call_together(client, system_prompt, prompt_text, b64_images, model, max_output_tokens, step_label)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Expected one of: {list(DEFAULT_MODEL)}")


def _get_evidence_page(result: str) -> Optional[int]:
    """Return the first non-null page from the evidence array of the first sample."""
    try:
        blob = _parse_json_blob(result)
        samples = blob.get("samples") or blob.get("records") or []
        if not samples:
            return None
        for ev in samples[0].get("evidence", []):
            if ev.get("page") is not None:
                return int(ev["page"])
        return None
    except Exception:
        return None


def _adjust_evidence_pages(result: str, offset: int) -> str:
    """Add *offset* to every non-null evidence[*].page value in a JSON result string."""
    if offset == 0:
        return result
    try:
        blob = _parse_json_blob(result)
        for sample in blob.get("samples") or blob.get("records") or []:
            for ev in sample.get("evidence", []):
                if ev.get("page") is not None:
                    ev["page"] = ev["page"] + offset
        return json.dumps(blob)
    except Exception:
        return result


# ---------------------------------------------------------------------------
# System prompts and evidence appendices
# ---------------------------------------------------------------------------

_EVIDENCE_APPENDIX_BASE = """
---
SUPPORTING EVIDENCE REQUIREMENT

Add an "evidence" key to every sample/record object in your output. Its value is a JSON array. Each array element must have exactly these four keys:

- "snippet": the exact verbatim text from the paper that supports the extraction (do not paraphrase — quote directly)
- "page": the sequential PDF page number (1 = first page provided). Do NOT use the journal or book page number printed in the document header/footer.
- "source": the table or figure identifier (e.g. "Table 2", "Figure 1A"), or null if not from a named element
- "field": a short label for what this evidence supports

Do NOT embed snippet/page/source inline with extracted values — keep all evidence in the "evidence" array only."""

EVIDENCE_APPENDIX_FACTORS = _EVIDENCE_APPENDIX_BASE + """

Coverage: include at least one entry for sample identification, one entry per factor column (citing the table containing the loadings), and one entry for the total number of factors.

Example:
"evidence": [
  {"snippet": "Table 2. Rotated factor matrix...", "page": 4, "source": "Table 2", "field": "F1 loadings"},
  {"snippet": "N = 147 undergraduate students", "page": 2, "source": null, "field": "sample identification"},
  {"snippet": "A two-factor solution was retained", "page": 3, "source": null, "field": "factor count"}
]"""

EVIDENCE_APPENDIX_CORRELATIONS = _EVIDENCE_APPENDIX_BASE + """

Coverage: include at least one entry for sample identification and one entry citing where the factor correlations were found.

Example:
"evidence": [
  {"snippet": "Table 3. Factor correlation matrix (Phi)", "page": 5, "source": "Table 3", "field": "factor correlations"},
  {"snippet": "N = 147 undergraduate students", "page": 2, "source": null, "field": "sample identification"}
]"""

EVIDENCE_APPENDIX_METADATA = _EVIDENCE_APPENDIX_BASE + """

Coverage: include one evidence entry per extracted field (e.g. n, age, country, rot). Use the field name as the "field" value. If a field is null because the information is absent, include the entry with snippet/source/page all set to null.

Example:
"evidence": [
  {"snippet": "The sample consisted of 312 undergraduate students", "page": 2, "source": null, "field": "n"},
  {"snippet": "mean age was 21.3 years", "page": 2, "source": null, "field": "age"},
  {"snippet": "promax rotation", "page": 3, "source": null, "field": "rot"}
]"""

FACTOR_SYSTEM = (
    "You are a data extraction assistant. "
    "The user will show you pages of an academic paper as images. "
    "Follow the extraction instructions exactly and output only valid JSON."
)

CORRELATION_SYSTEM = (
    "You are a data extraction assistant. "
    "The user will show you pages of an academic paper as images. "
    "Follow the extraction instructions exactly and output only valid JSON."
)

METADATA_SYSTEM = (
    "You are a data extraction assistant. "
    "The user will show you pages of an academic paper as images plus previously "
    "extracted JSON results. Follow the extraction instructions exactly and output "
    "only valid JSON."
)


# ---------------------------------------------------------------------------
# Per-task extraction functions (chunk-aware)
# ---------------------------------------------------------------------------

def _extract_factors(
    client,
    provider: str,
    image_chunks: list[list[str]],
    prompt_text: str,
    model: str,
    max_output_tokens: int,
) -> str:
    """Run factor-loadings extraction across image chunks; merge results across all chunks."""
    full_prompt = prompt_text + EVIDENCE_APPENDIX_FACTORS
    results: list[str] = []
    offsets: list[int] = []
    offset = 0
    for idx, chunk in enumerate(image_chunks):
        label = f"factors-chunk{idx + 1}/{len(image_chunks)}"
        if len(image_chunks) > 1:
            print(f"  chunk {idx + 1}/{len(image_chunks)} ({len(chunk)} pages)")
        results.append(_call_model(client, provider, FACTOR_SYSTEM, full_prompt, chunk, model, max_output_tokens, label))
        offsets.append(offset)
        offset += len(chunk)

    if len(image_chunks) == 1:
        return _adjust_evidence_pages(results[0], 0)
    return _merge_samples(results, offsets, "factor_loadings")


def _extract_correlations(
    client,
    provider: str,
    image_chunks: list[list[str]],
    prompt_text: str,
    model: str,
    max_output_tokens: int,
) -> str:
    """Run correlation extraction across image chunks; merge results across all chunks."""
    full_prompt = prompt_text + EVIDENCE_APPENDIX_CORRELATIONS
    results: list[str] = []
    offsets: list[int] = []
    offset = 0
    for idx, chunk in enumerate(image_chunks):
        label = f"correlations-chunk{idx + 1}/{len(image_chunks)}"
        if len(image_chunks) > 1:
            print(f"  chunk {idx + 1}/{len(image_chunks)} ({len(chunk)} pages)")
        results.append(_call_model(client, provider, CORRELATION_SYSTEM, full_prompt, chunk, model, max_output_tokens, label))
        offsets.append(offset)
        offset += len(chunk)

    if len(image_chunks) == 1:
        return _adjust_evidence_pages(results[0], 0)
    return _merge_samples(results, offsets, "factor_correlations")


def _extract_metadata(
    client,
    provider: str,
    image_chunks: list[list[str]],
    factor_result: str,
    correlation_result: str,
    prompt_text: str,
    model: str,
    max_output_tokens: int,
) -> str:
    """Run metadata extraction across all image chunks and merge results."""
    user_text = (
        f"{prompt_text}\n\n"
        "=== EXTRACTED FACTOR LOADINGS ===\n"
        f"{factor_result}\n\n"
        "=== EXTRACTED CORRELATIONS ===\n"
        f"{correlation_result}\n\n"
        "Now look at the paper pages below and extract the metadata."
        + EVIDENCE_APPENDIX_METADATA
    )
    results: list[str] = []
    offsets: list[int] = []
    offset = 0
    for idx, chunk in enumerate(image_chunks):
        label = f"metadata-chunk{idx + 1}/{len(image_chunks)}"
        if len(image_chunks) > 1:
            print(f"  chunk {idx + 1}/{len(image_chunks)} ({len(chunk)} pages)")
        results.append(_call_model(client, provider, METADATA_SYSTEM, user_text, chunk, model, max_output_tokens, label))
        offsets.append(offset)
        offset += len(chunk)

    if len(results) == 1:
        return _adjust_evidence_pages(results[0], 0)
    return _merge_metadata(results, offsets)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    pdf_name: str,
    force: bool = False,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "openai",
    dpi: int = 200,
    max_output_tokens: int = 16000,
    chunk_size: int = 15,
) -> list[Path]:
    """Run single-step MASEM extraction on a PDF using vision LLM.

    Args:
        pdf_name:         PDF file name (e.g., wise2000.pdf) or full path.
        force:            Re-run even if output files already exist.
        dataset:          Dataset label (e.g., "tuning"). Output goes to <dataset>_1step/.
        model:            Model name. Defaults to the provider's recommended model.
        provider:         API provider: "openai", "cohere", or "gemini".
        dpi:              Resolution for PDF-to-image conversion.
        max_output_tokens: Token budget per API call.
        chunk_size:       Max pages per API call. Papers longer than this are split.

    Returns:
        List of Paths to the output JSON files that were written.
    """
    load_dotenv()

    if provider not in DEFAULT_MODEL:
        raise ValueError(f"Unknown provider '{provider}'. Expected one of: {list(DEFAULT_MODEL)}")
    if model is None:
        model = DEFAULT_MODEL[provider]

    # Resolve PDF path
    pdf_path = Path(pdf_name)
    if not pdf_path.exists():
        candidates = []
        if dataset:
            candidates.append(ROOT / "data" / "intermediate_data" / dataset / pdf_path.name)
        candidates.append(ROOT / "data" / "intermediate_data" / pdf_path.name)
        for c in candidates:
            if c.exists():
                pdf_path = c
                break
        else:
            raise FileNotFoundError(f"PDF not found: {pdf_name}")

    study_name = pdf_path.stem
    tag = f"{dataset}_{provider}" if dataset else provider

    # Output paths
    factor_dir = Path("data/extraction_factors") / tag
    correlation_dir = Path("data/extraction_correlations") / tag
    metadata_dir = Path("data/extraction_metadata") / tag
    factor_path = factor_dir / f"{study_name}.json"
    correlation_path = correlation_dir / f"{study_name}.json"
    metadata_path = metadata_dir / f"{study_name}.json"

    if not force and factor_path.exists() and correlation_path.exists() and metadata_path.exists():
        print(f"Skipping {study_name}: all 1-step outputs already exist")
        return [factor_path, correlation_path, metadata_path]

    # Build client for the selected provider
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        client = OpenAI(api_key=api_key)
    elif provider == "cohere":
        if cohere_sdk is None:
            raise ImportError("cohere package not installed. Run: pip install cohere")
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment.")
        client = cohere_sdk.ClientV2(api_key=api_key)
    elif provider == "gemini":
        if genai_sdk is None:
            raise ImportError(
                "google-genai package not installed. Run: uv pip install google-genai"
            )
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
        client = genai_sdk.Client(api_key=api_key)
    elif provider == "together":
        if together_sdk is None:
            raise ImportError("together package not installed. Run: pip install together")
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment.")
        client = together_sdk(api_key=api_key)

    # Load prompts
    factor_prompt = _load_prompt_text("prompt_factor_loadings.txt")
    correlation_prompt = _load_prompt_text("prompt_correlations.txt")
    metadata_prompt = _load_prompt_text("prompt_metadata.txt")

    # Convert PDF to images once, then chunk
    print(f"\n{'=' * 60}")
    print(f"1-step MASEM extraction: {study_name}  (provider={provider}, model={model})")
    print(f"{'=' * 60}")
    print(f"Converting PDF to images at {dpi} DPI...")
    b64_images = _pdf_to_base64_images(pdf_path, dpi=dpi)
    if len(b64_images) > 50:
        print(
            f"\nWARNING: {study_name} has {len(b64_images)} pages (limit: 50). "
            "Please select a shorter version of this paper and re-run. Skipping.\n"
        )
        return []
    provider_max = MAX_IMAGES_PER_CALL.get(provider, chunk_size)
    if chunk_size > provider_max:
        print(f"  Note: clamping chunk_size from {chunk_size} to {provider_max} (provider limit)")
        chunk_size = provider_max
    image_chunks = _chunk_images(b64_images, chunk_size=chunk_size)
    print(f"  {len(b64_images)} page(s) → {len(image_chunks)} chunk(s) of up to {chunk_size}")

    output_paths: list[Path] = []

    def _save(path: Path, raw_result: str, prompt_used: str) -> None:
        """Parse raw model output and save in the standard envelope format."""
        try:
            blob = _parse_json_blob(raw_result)
            entries = blob.get("samples") or blob.get("records") or []
        except Exception:
            entries = []
        envelope = {
            "filename": pdf_path.name,
            "pages_processed": len(b64_images),
            "model": model,
            "prompt": prompt_used,
            "entries": entries,
            "human_overrides": [],
            "original_model_response": raw_result,
        }
        path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    def _load_entries(path: Path) -> str:
        """Read entries from an existing envelope file and return as a JSON string."""
        data = json.loads(path.read_text(encoding="utf-8"))
        # Re-wrap entries in samples/records so downstream merge functions still work
        entries = data.get("entries", [])
        return json.dumps({"samples": entries})

    # ---- Task 1: Factor loadings ----
    factor_prompt_full = factor_prompt + EVIDENCE_APPENDIX_FACTORS
    if force or not factor_path.exists():
        print("\nTask 1: Factor loadings extraction...")
        factor_result = _extract_factors(
            client, provider, image_chunks, factor_prompt, model, max_output_tokens
        )
        factor_dir.mkdir(parents=True, exist_ok=True)
        _save(factor_path, factor_result, factor_prompt_full)
        print(f"  ✓ Saved to {factor_path}")
    else:
        print(f"\nTask 1: Skipping factor loadings (already exists at {factor_path})")
        factor_result = _load_entries(factor_path)
    output_paths.append(factor_path)

    # ---- Task 2: Correlations ----
    correlation_prompt_full = correlation_prompt + EVIDENCE_APPENDIX_CORRELATIONS
    if force or not correlation_path.exists():
        print("\nTask 2: Correlations extraction...")
        correlation_result = _extract_correlations(
            client, provider, image_chunks, correlation_prompt, model, max_output_tokens
        )
        correlation_dir.mkdir(parents=True, exist_ok=True)
        _save(correlation_path, correlation_result, correlation_prompt_full)
        print(f"  ✓ Saved to {correlation_path}")
    else:
        print(f"\nTask 2: Skipping correlations (already exists at {correlation_path})")
        correlation_result = _load_entries(correlation_path)
    output_paths.append(correlation_path)

    # ---- Task 3: Metadata ----
    metadata_prompt_full = (
        metadata_prompt + "\n\n"
        "=== EXTRACTED FACTOR LOADINGS ===\n"
        f"{factor_result}\n\n"
        "=== EXTRACTED CORRELATIONS ===\n"
        f"{correlation_result}\n\n"
        "Now look at the paper pages below and extract the metadata."
        + EVIDENCE_APPENDIX_METADATA
    )
    if force or not metadata_path.exists():
        print("\nTask 3: Metadata extraction...")
        metadata_result = _extract_metadata(
            client, provider, image_chunks, factor_result, correlation_result,
            metadata_prompt, model, max_output_tokens,
        )
        metadata_dir.mkdir(parents=True, exist_ok=True)
        _save(metadata_path, metadata_result, metadata_prompt_full)
        print(f"  ✓ Saved to {metadata_path}")
    else:
        print(f"\nTask 3: Skipping metadata (already exists at {metadata_path})")
    output_paths.append(metadata_path)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(output_paths)} output(s) written for {study_name}")
    print(f"{'=' * 60}\n")

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-step MASEM extraction using GPT vision (no intermediate markdown).\n"
            "Runs factor loadings, correlations, and metadata extraction using PDF page\n"
            "images sent directly to GPT. Papers longer than --chunk-size pages are\n"
            "split into chunks; results are merged automatically.\n\n"
            "Outputs land in <dataset>_1step/ folders:\n"
            "  data/extraction_factors/<dataset>_1step/<study>.json\n"
            "  data/extraction_correlations/<dataset>_1step/<study>.json\n"
            "  data/extraction_metadata/<dataset>_1step/<study>.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf_name", help="PDF file name or full path (e.g., wise2000.pdf).")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs already exist.")
    parser.add_argument("--dataset", help="Dataset label (e.g., tuning). Output goes to <dataset>_1step/.")
    parser.add_argument("--provider", default="openai", choices=["openai", "cohere", "gemini", "together"],
                        help="API provider to use (default: openai).")
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults to provider's recommended model "
                             f"({', '.join(f'{k}: {v}' for k, v in DEFAULT_MODEL.items())}).")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF-to-image conversion (default: 200).")
    parser.add_argument("--max-output-tokens", type=int, default=16000, dest="max_output_tokens",
                        help="Max output tokens per API call (default: 8000).")
    parser.add_argument("--chunk-size", type=int, default=15, dest="chunk_size",
                        help="Max pages per API call (default: 15). Longer papers are split into chunks.")
    args = parser.parse_args()
    run(
        pdf_name=args.pdf_name,
        force=args.force,
        dataset=args.dataset,
        model=args.model,
        provider=args.provider,
        dpi=args.dpi,
        max_output_tokens=args.max_output_tokens,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
