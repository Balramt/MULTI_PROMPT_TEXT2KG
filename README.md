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

The `data/` directory contains all datasets, ontologies, training data, and evaluation inputs required to reproduce the experiments.

## 1️⃣ Input Text  
🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/input_text  

Contains the raw input sentences for both **DBpedia–WebNLG** and **Wikidata–TekGen** datasets used during inference and evaluation.

---

## 2️⃣ Ground Truth  
🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ground_truth  

Gold standard SPO triples for evaluation. These are used to compute **precision, recall, and F1-score**.

---

## 3️⃣ Few-Shot Examples  
🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/fewshots_example  

Example triples used inside prompts to guide the model during extraction (few-shot prompting).

---

## 4️⃣ Ontology Files  
🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/input/ontology/old_ontology  

Domain-specific ontology schemas containing:

- Concept sets  
- Relation definitions  
- Domain–range constraints  

These are injected into prompts to enforce ontology-aware extraction.

---

## 5️⃣ Training Data  
🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/data/train_data  

Contains the combined and enriched training data used for fine-tuning the **LLaMA-3** model.

---

# 🧠 src/ – Main Source Directory  

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/tree/main/src  

The `src/` directory contains the complete implementation of:

- Synthetic data generation  
- Model fine-tuning  
- Multi-prompt triple extraction  
- Evaluator logic  
- Performance and hallucination evaluation  

Below is a breakdown of each component.

---

## 🔹 1️⃣ Synthetic Data Generation  

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Synthetic_train_data_generation_7B.py  

Generates additional ontology-aligned triples for **Wikidata–TekGen** training data using a larger LLM.  
The script preserves seed triples and filters outputs based on ontology constraints to reduce noise.

---

## 🔹 2️⃣ LLaMA-3 Fine-Tuning  

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Llama_finetuned.py  

Performs supervised fine-tuning (SFT) of **LLaMA-3-8B-Instruct** using LoRA/QLoRA.  
The model learns structured JSON SPO output and ontology-aware relation naming.

---

# 🔁 Multi-Prompt Triple Extraction Modules

## 1️⃣ Structured Multi-Step Reasoning (ToT-Based Extractor)

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/Llama_finetuned.py  

Implements a Tree-of-Thoughts (ToT) style extraction using depth-first search, state scoring, and pruning under ontology constraints.

---

## 2️⃣ Ontology-Constrained OpenIE Prompt

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/Open_IE_prompt.py  

Single-pass ontology-aware extraction enforcing domain–range constraints, semantic types, and textual evidence spans.

---

## 3️⃣ General Ontology-Aware Extraction Prompt

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/multi_prompt_extractor/general_extraction_prompt.py  

Lightweight SPO extraction with minimal structural constraints.  
Provides high recall and complementary coverage to stricter prompts.

---

# 🧠 Evaluator Module

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/evaluator.py  

Implements hierarchical verification using:

- **Rule A – Cross-Prompt Consensus**
- **Rule B – Evidence-Based Validation**
- **Rule C – Similarity-Based Filtering**

The evaluator merges and filters triples to reduce hallucinations and enforce schema compliance.

---

# 📊 Evaluation Scripts

🔗 https://github.com/Balramt/MULTI_PROMPT_TEXT2KG/blob/main/src/metrics_evaluation.py  

Computes:

- Precision  
- Recall  
- F1-score  
- Ontology Conformance (**OC ↑**)  
- Subject Hallucination (**SH ↓**)  
- Relation Hallucination (**RH ↓**)  
- Object Hallucination (**OH ↓**)  

All reported metrics are calculated **after evaluator filtering**.
