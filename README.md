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

### 📂 [`input_text`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text)

Contains raw input sentences for both **DBpedia–WebNLG** and **Wikidata–TekGen** used during inference and evaluation.

- [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text/dbpedia)  
- [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text/wikidata)  

---

### 📂 [`ground_truth`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth)

Gold standard SPO triples used to compute **Precision, Recall, and F1-score**.

- [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth/dbpedia)  
- [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth/wikidata)  

---

### 📂 [`fewshots_example`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example)

Few-shot examples injected into prompts to guide ontology-aligned triple extraction.

- [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example/dbpedia)  
- [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example/wikidata)  

---

### 📂 [`ontology`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology)

Domain-specific ontology schemas including:

- Concept definitions  
- Relation signatures  
- Domain–range constraints  

These are injected directly into prompts to enforce schema compliance.

- [**DBpedia Ontologies**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology/dbpedia)  
- [**Wikidata Ontologies**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology/wikidata)  

---

### 📂 [`train_data`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data)

Combined and enriched training dataset used for **LLaMA-3 fine-tuning**.

- [**DBpedia Training Data**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/dbpedia)

- **Wikidata Training Data (Synthetic Enrichment Pipeline)**  

  - [Input Wikidata Train Data](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/wikidata/synthetic_train_data/wikidata_input_train)  
  - [Generated & Filtered Output Train Data Used for Fine-Tuning](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/wikidata/synthetic_train_data/wikidata_output_train)

The Wikidata dataset is synthetically enriched to compensate for incomplete distant supervision. The filtered output data is directly used during LLaMA-3 supervised fine-tuning.

---

# 📤 Output Data

This section contains all generated outputs from the multi-prompt extraction pipeline, evaluator filtering stage, and final evaluation metrics.

## 🔁 Multi-Prompt Extraction Outputs

Each prompting strategy generates an independent candidate triple set before evaluator filtering.

### 📂 [`TOT_dfs`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/TOT_dfs)

Structured **Tree-of-Thoughts (ToT)-based depth-first search extraction** outputs.

---

### 📂 [`Open_IE_prompt`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/Open_IE_prompt)

Outputs from the **Ontology-Constrained Open Information Extraction** prompt.

---

### 📂 [`general_extraction_prompt`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/general_extraction_prompt)

Outputs from the **General Ontology-Aware Extraction Prompt**.

---

## 🧠 Evaluator-Filtered Outputs

### 📂 [`evaluator_filtered_output`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/evaluator_filtered_output)

Final merged and filtered triple sets after applying:

- Rule A – Cross-Prompt Consensus  
- Rule B – Evidence-Based Validation  
- Rule C – Similarity-Based Filtering  

---

## 📊 Evaluation Results

### 📂 [`metrics_evaluation`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation)

Dataset-wise results:

- [**DBpedia–WebNLG**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation/dbpedia)  
- [**Wikidata–TekGen**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation/wikidata)

Reported Metrics:

- Precision (P)  
- Recall (R)  
- F1-score (F1)  
- Ontology Conformance (OC ↑)  
- Subject Hallucination (SH ↓)  
- Relation Hallucination (RH ↓)  
- Object Hallucination (OH ↓)  

---

# 🧠 Source Code (`src/`)

### 📂 [`src`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/src)

Contains:

- Synthetic data generation  
- Model fine-tuning  
- Multi-prompt extraction  
- Evaluator logic  
- Evaluation metrics  

---

## 🔹 Data Preparation & Training

### 📂 [`Synthetic_train_data_generation_7B.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Synthetic_train_data_generation_7B.py)

Generates ontology-filtered synthetic triples for **Wikidata–TekGen**.

---

### 📂 [`Llama_finetuned.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Llama_finetuned.py)

Performs supervised fine-tuning (SFT) of **LLaMA-3-8B-Instruct** using LoRA/QLoRA.

---

## 🔁 Multi-Prompt Extraction Modules

### 📂 [`Open_IE_prompt.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/Open_IE_prompt.py)

Ontology-constrained OpenIE extraction.

---

### 📂 [`general_extraction_prompt.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/general_extraction_prompt.py)

Lightweight ontology-aware SPO extraction.

---

### 📂 [`evaluator.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/evaluator.py)

Hierarchical triple verification engine.

---

# 📁 Project Directory Structure

MULTI_PROMPT_TEXT2KG/
│
├── data/
│ ├── input/
│ │ ├── input_text/
│ │ ├── ground_truth/
│ │ ├── fewshots_example/
│ │ └── ontology/
│ │
│ ├── train_data/
│ │ ├── dbpedia/
│ │ └── wikidata/
│ │
│ └── output/
│ ├── multi_step_prompts/
│ │ ├── TOT_dfs/
│ │ ├── Open_IE_prompt/
│ │ └── general_extraction_prompt/
│ │
│ ├── evaluator_filtered_output/
│ └── metrics_evaluation/
│
├── src/
│ ├── Synthetic_train_data_generation_7B.py
│ ├── Llama_finetuned.py
│ ├── multi_prompt_extractor/
│ │ ├── Open_IE_prompt.py
│ │ └── general_extraction_prompt.py
│ └── evaluator.py
│
└── README.md


---
