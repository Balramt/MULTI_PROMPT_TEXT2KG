# MULTI_PROMPT_TEXT2KG

This repository contains the official implementation of our research framework for **Ontology-Aware Knowledge Graph Construction from Unstructured Text using Large Language Models (LLMs)**.

The system combines multiple prompting strategies with a hierarchical evaluator to improve extraction accuracy while mitigating hallucinations. It is designed and evaluated on the **Text2KGBench** benchmark.

---

# 🚀 Overview

Knowledge Graph (KG) construction from natural language is challenging due to:

- Incomplete supervision  
- Ontology constraints  
- Hallucinated entities and relations  
- Inconsistent triple formatting  

To address these issues, we propose a **multi-prompt ensemble framework** consisting of:

- 🔁 **Structured Multi-Step Reasoning (ToT-based)**
- 📚 **Ontology-Constrained OpenIE Prompt**
- ⚡ **General Ontology-Aware Extraction Prompt**
- 🧠 **Hierarchical Evaluator (Rules A–C)**

The evaluator filters candidate triples using cross-prompt agreement, explicit evidence scoring, and textual similarity measures to reduce hallucinations and enforce schema compliance.

---

# 📂 Data Directory

## 🟢 Input Data

### 📂 `data/input/input_text/`

Raw input sentences for:

- **DBpedia–WebNLG**
- **Wikidata–TekGen**

Subdirectories:
- `dbpedia/`
- `wikidata/`

---

### 📂 `data/input/ground_truth/`

Gold standard SPO triples used to compute:

- Precision
- Recall
- F1-score

Subdirectories:
- `dbpedia/`
- `wikidata/`

---

### 📂 `data/input/fewshots_example/`

Few-shot examples injected into prompts to guide ontology-aligned extraction.

Subdirectories:
- `dbpedia/`
- `wikidata/`

---

### 📂 `data/input/ontology/old_ontology/`

Domain-specific ontology schemas including:

- Concept definitions  
- Relation signatures  
- Domain–range constraints  

Subdirectories:
- `dbpedia/`
- `wikidata/`

---

### 📂 `data/train_data/`

Training datasets used for **LLaMA-3 fine-tuning**.

- `dbpedia/`
- `wikidata/`

Wikidata includes a synthetic enrichment pipeline:

- `synthetic_train_data/wikidata_input_train/`
- `synthetic_train_data/wikidata_output_train/`

The filtered output is directly used for supervised fine-tuning.

---

# 📤 Output Data

## 🔁 Multi-Prompt Extraction Outputs

Located in:

`data/output/multi_step_prompts/`

- `TOT_dfs/` → Tree-of-Thoughts structured extraction  
- `Open_IE_prompt/` → Ontology-constrained OpenIE  
- `general_extraction_prompt/` → Lightweight SPO extraction  

---

## 🧠 Evaluator-Filtered Outputs

`data/output/evaluator_filtered_output/`

Final merged triple sets after applying:

- Rule A – Cross-Prompt Consensus  
- Rule B – Evidence-Based Validation  
- Rule C – Similarity-Based Filtering  

---

## 📊 Evaluation Results

`data/output/metrics_evaluation/`

Contains dataset-wise evaluation:

- `dbpedia/`
- `wikidata/`

Metrics reported:

- Precision (P)
- Recall (R)
- F1-score (F1)
- Ontology Conformance (OC ↑)
- Subject Hallucination (SH ↓)
- Relation Hallucination (RH ↓)
- Object Hallucination (OH ↓)

---

# 🧠 Source Code (`src/`)

Main implementation directory.

Contains:

- Synthetic data generation  
- Model fine-tuning  
- Multi-prompt extraction  
- Evaluator logic  
- Evaluation metrics  

---

## 🔹 Data Preparation & Training

### `Synthetic_train_data_generation_7B.py`

Generates ontology-filtered synthetic triples for Wikidata–TekGen.

---

### `Llama_finetuned.py`

Supervised fine-tuning (SFT) of LLaMA-3-8B-Instruct using LoRA/QLoRA.

---

## 🔁 Multi-Prompt Extraction Modules

Located in:

`src/multi_prompt_extractor/`

- `Open_IE_prompt.py`
- `general_extraction_prompt.py`

---

### `evaluator.py`

Implements hierarchical triple verification:

- Cross-Prompt Consensus
- Evidence-Based Validation
- Similarity-Based Filtering

---

# 📁 Project Directory Structure

```text
MULTI_PROMPT_TEXT2KG/
│
├── data/
│   ├── input/
│   │   ├── input_text/
│   │   │   ├── dbpedia/
│   │   │   └── wikidata/
│   │   ├── ground_truth/
│   │   │   ├── dbpedia/
│   │   │   └── wikidata/
│   │   ├── fewshots_example/
│   │   │   ├── dbpedia/
│   │   │   └── wikidata/
│   │   └── ontology/
│   │       └── old_ontology/
│   │           ├── dbpedia/
│   │           └── wikidata/
│   │
│   ├── train_data/
│   │   ├── dbpedia/
│   │   └── wikidata/
│   │       └── synthetic_train_data/
│   │           ├── wikidata_input_train/
│   │           └── wikidata_output_train/
│   │
│   └── output/
│       ├── multi_step_prompts/
│       │   ├── TOT_dfs/
│       │   ├── Open_IE_prompt/
│       │   └── general_extraction_prompt/
│       │
│       ├── evaluator_filtered_output/
│       └── metrics_evaluation/
│           ├── dbpedia/
│           └── wikidata/
│
├── src/
│   ├── Synthetic_train_data_generation_7B.py
│   ├── Llama_finetuned.py
│   ├── evaluator.py
│   └── multi_prompt_extractor/
│       ├── Open_IE_prompt.py
│       └── general_extraction_prompt.py
│
└── README.md
