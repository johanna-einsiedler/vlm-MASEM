"""
tune_val_eval_split.py

Threefold split for papers in 'data/intermediate_data':
(1) 10 tuning papers
(2) 10 validation papers
(3) remaining as evaluation

Usage:
    python tune_val_eval_split.py
"""

import os
import random
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "intermediate_data")
TUNING_DIR = os.path.join(DATA_DIR, "tuning")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
TUNING_SIZE = 10
VALIDATION_SIZE = 10

os.makedirs(TUNING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

word_exts = (".doc", ".docx")
word_docs = [
    f
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(word_exts) and os.path.isfile(os.path.join(DATA_DIR, f))
]
if word_docs:
    print(
        f"Warning: {len(word_docs)} Word document(s) found in 'data/intermediate_data' and not included in the split:"
    )
    for doc in word_docs:
        print(f"  - {doc}")

pdfs = [
    f
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(DATA_DIR, f))
]

if len(pdfs) < TUNING_SIZE + VALIDATION_SIZE:
    raise ValueError(
        f"Not enough PDF files in 'data/intermediate_data' to sample {TUNING_SIZE + VALIDATION_SIZE} (found {len(pdfs)})"
    )

all_pdfs = set(pdfs)
tuning_pdfs = set(random.sample(pdfs, TUNING_SIZE))
remaining_pdfs = list(all_pdfs - tuning_pdfs)
validation_pdfs = set(random.sample(remaining_pdfs, VALIDATION_SIZE))
eval_pdfs = all_pdfs - tuning_pdfs - validation_pdfs

for pdf in tuning_pdfs:
    shutil.move(os.path.join(DATA_DIR, pdf), os.path.join(TUNING_DIR, pdf))
for pdf in validation_pdfs:
    shutil.move(os.path.join(DATA_DIR, pdf), os.path.join(VALIDATION_DIR, pdf))
for pdf in eval_pdfs:
    shutil.move(os.path.join(DATA_DIR, pdf), os.path.join(EVAL_DIR, pdf))

print(f"Tuning set: {len(tuning_pdfs)} studies")
print(f"Validation set: {len(validation_pdfs)} studies")
print(f"Evaluation set: {len(eval_pdfs)} studies")
if word_docs:
    print(f"Non-PDF (Word) docs not assigned: {len(word_docs)}")
