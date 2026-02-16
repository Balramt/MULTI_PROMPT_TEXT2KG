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

# Input data

### 📂 [`input_text`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text)
Contains raw input sentences for both **DBpedia–WebNLG** and **Wikidata–TekGen** used during inference and evaluation.

1. [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text/dbpedia)

2. [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text/wikidata)

---

### 📂 [`ground_truth`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth)
Gold standard SPO triples used to compute **Precision, Recall, and F1-score**.

1. [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth/dbpedia)

2. [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth/wikidata)

---

### 📂 [`fewshots_example`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example)
Few-shot examples injected into prompts to guide ontology-aligned triple extraction.

1. [**DBpedia**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example/dbpedia)

2. [**Wikidata**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example/wikidata)

---

### 📂 [`ontology`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology)
Domain-specific ontology schemas including:

- Concept definitions  
- Relation signatures  
- Domain–range constraints  

These are directly injected into prompts to enforce schema compliance.

1. [**DBpedia Ontologies**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology/dbpedia)

2. [**Wikidata Ontologies**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology/wikidata)

---

### 📂 [`train_data`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data)
Combined and enriched training dataset used for **LLaMA-3 fine-tuning**.

1. [**DBpedia Training Data**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/dbpedia)

2. **Wikidata Training Data (Synthetic Enrichment Pipeline)**  

   - [**Input Wikidata Train Data** ](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/wikidata/synthetic_train_data/wikidata_input_train)

   - [**Generated & Filtered Output Train Data Used for Fine-Tuning**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data/wikidata/synthetic_train_data/wikidata_output_train)

   The Wikidata dataset is synthetically enriched to compensate for incomplete distant supervision.  
   The filtered output data is directly used during LLaMA-3 supervised fine-tuning.

# 📤 Output Data

This section contains all generated outputs from the multi-prompt extraction pipeline, evaluator filtering stage, and final evaluation metrics.

---

## 🔁 Multi-Prompt Extraction Outputs

Each prompting strategy generates an independent candidate triple set before evaluator filtering.

### 🔹 [`TOT_dfs`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/TOT_dfs)
Structured **Tree-of-Thoughts (ToT)-based depth-first search extraction** outputs.  
Includes intermediate reasoning states and final candidate triples generated under ontology constraints.

---

### 🔹 [`Open_IE_prompt`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/Open_IE_prompt)
Outputs from the **Ontology-Constrained Open Information Extraction** prompt.  
Single-pass extraction enforcing domain–range constraints, semantic typing, and evidence spans.

---

### 🔹 [`general_extraction_prompt`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/multi_step_prompts/general_extraction_prompt)
Outputs from the **General Ontology-Aware Extraction Prompt**.  
Lightweight high-recall SPO extraction with minimal structural constraints.

---

## 🧠 Evaluator-Filtered Outputs

### 🔹 [`evaluator_filtered_output`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/evaluator_filtered_output)

Final merged and filtered triple sets after applying:

- Rule A – Cross-Prompt Consensus  
- Rule B – Evidence-Based Validation  
- Rule C – Similarity-Based Filtering  

All reported experimental results are computed using these filtered outputs.

---

## 📊 Evaluation Results

### 🔹 [`metrics_evaluation`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation)

Contains performance and hallucination metrics computed after evaluator filtering.

### 📁 Dataset-wise Results

1. [**DBpedia–WebNLG**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation/dbpedia)

2. [**Wikidata–TekGen**](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/output/metrics_evaluation/wikidata)

---

### 📈 Reported Metrics

- Precision (P)  
- Recall (R)  
- F1-score (F1)  
- Ontology Conformance (OC ↑)  
- Subject Hallucination (SH ↓)  
- Relation Hallucination (RH ↓)  
- Object Hallucination (OH ↓)  

All metrics are computed on the **evaluator-filtered triple sets**.



# 🧠 Source Code (`src/`)

### 📂 [`src`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/src)
Contains the full implementation of:

- Synthetic data generation  
- Model fine-tuning  
- Multi-prompt extraction  
- Evaluator logic  
- Evaluation metrics  

---

## 🔹 Data Preparation & Training

### 📂 [`Synthetic_train_data_generation_7B.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Synthetic_train_data_generation_7B.py)
Generates ontology-filtered synthetic triples for **Wikidata–TekGen** using a larger LLM.  
Preserves seed triples and removes out-of-schema relations to improve training quality.

---

### 📂 [`Llama_finetuned.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Llama_finetuned.py)
Performs supervised fine-tuning (SFT) of **LLaMA-3-8B-Instruct** using LoRA/QLoRA.  
Trains the model to generate structured JSON SPO triples with ontology-aware relation naming.

---

# 🔁 Multi-Prompt Extraction Modules

### 📂 [`ToT-Based Structured Extractor`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Llama_finetuned.py)
Implements Tree-of-Thoughts style extraction with depth-first search, state scoring, and pruning under ontology constraints.

---

### 📂 [`Open_IE_prompt.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/Open_IE_prompt.py)
Single-pass ontology-constrained OpenIE extraction enforcing domain–range rules, semantic typing, and evidence spans.

---

### 📂 [`general_extraction_prompt.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/general_extraction_prompt.py)
Lightweight ontology-aware SPO extraction with minimal structural constraints, providing high recall and complementary coverage.

---

# 🧠 Evaluator

### 📂 [`evaluator.py`](https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/evaluator.py)
Implements hierarchical triple verification:

- **Rule A – Cross-Prompt Consensus**
- **Rule B – Evidence-Based Validation**
- **Rule C – Similarity-Based Filtering**

Merges and filters candidate triples to reduce hallucinations and ensure ontology consistency.

---
