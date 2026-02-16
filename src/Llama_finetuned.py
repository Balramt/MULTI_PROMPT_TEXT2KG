import os
import json
from typing import Dict, Any, List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def run_prompt4_qlora_finetune():
    DBPEDIA_TRAIN_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/train_data/dbpedia"
    DBPEDIA_ONTOLOGY_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/ontology/old_ontology/dbpedia"

    WIKIDATA_TRAIN_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/train_data/wikidata/synthetic_train_data/wikidata_output_train"
    WIKIDATA_ONTOLOGY_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/ontology/old_ontology/wikidata"

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    OUTPUT_DIR = "upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/finetune_models/llama3_prompt4_finetuned_ontologyAware"

    DBPEDIA_ONTOLOGIES = [
        "ont_12_monument",
        "ont_1_university",
        "ont_2_musicalwork",
        "ont_3_airport",
        "ont_4_building",
        "ont_5_athlete",
        "ont_6_politician",
        "ont_7_company",
        "ont_8_celestialbody",
        "ont_9_astronaut",
        "ont_10_comicscharacter",
        "ont_11_meanoftransportation",
        "ont_13_food",
        "ont_14_writtenwork",
        "ont_15_sportsteam",
        "ont_16_city",
        "ont_17_artist",
        "ont_18_scientist",
        "ont_19_film",
    ]

    WIKIDATA_ONTOLOGIES = [
        "ont_1_movie",
        "ont_2_music",
        "ont_3_sport",
        "ont_4_book",
        "ont_5_military",
        "ont_6_computer",
        "ont_7_space",
        "ont_8_politics",
        "ont_9_nature",
        "ont_10_culture",
    ]
    WIKIDATA_ONTOLOGIES = list(dict.fromkeys(WIKIDATA_ONTOLOGIES))

    VERBOSE = True
    MAX_EXAMPLES = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    def clean_label(label: str) -> str:
        if label is None:
            return ""
        s = str(label).strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        s = s.replace("_", " ")
        return s

    def load_ontology(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            ont = json.load(f)
        return ont

    def ontology_to_text_dbpedia(ontology: Dict[str, Any]) -> Dict[str, str]:
        concepts = ontology.get("concepts", [])
        relations = ontology.get("relations", [])
        concept_labels = [c.get("label", "").strip() for c in concepts if c.get("label")]
        concepts_text = ", ".join(concept_labels)
        rel_lines = []
        for r in relations:
            name = r.get("label") or r.get("name") or ""
            dom = r.get("domain", "")
            ran = r.get("range", "")
            if name:
                if dom or ran:
                    rel_lines.append(f"- {name}({dom},{ran})")
                else:
                    rel_lines.append(f"- {name}")
        relations_text = "\n".join(rel_lines)
        return {"concepts_text": concepts_text, "relations_text": relations_text}

    def ontology_to_text_wikidata(ontology: Dict[str, Any]) -> Dict[str, str]:
        concepts = ontology.get("concepts", [])
        relations = ontology.get("relations", [])
        qid_to_label: Dict[str, str] = {}
        for c in concepts:
            qid = c.get("qid")
            label = c.get("label", "").strip()
            if qid and label:
                qid_to_label[qid] = label
        concept_labels = [c.get("label", "").strip() for c in concepts if c.get("label")]
        concepts_text = ", ".join(concept_labels)
        rel_lines = []
        for r in relations:
            name = r.get("label") or r.get("name") or ""
            dom_q = r.get("domain", "")
            ran_q = r.get("range", "")
            dom_label = qid_to_label.get(dom_q, dom_q)
            ran_label = qid_to_label.get(ran_q, ran_q)
            if name:
                if dom_label or ran_label:
                    rel_lines.append(f"- {name}({dom_label},{ran_label})")
                else:
                    rel_lines.append(f"- {name}")
        relations_text = "\n".join(rel_lines)
        return {"concepts_text": concepts_text, "relations_text": relations_text}

    SYSTEM_PROMPT_P4 = (
        "You are an expert information extraction model for knowledge graph construction. "
        "Given an ontology and a short text, you must extract all clearly expressed "
        "subject–predicate–object (SPO) triples. Use ontology relation names whenever possible. "
        "Your answer must be valid JSON only."
    )

    def build_prompt4_user_text(ontology_text: Dict[str, str], sentence: str) -> str:
        concepts_text = ontology_text["concepts_text"]
        relations_text = ontology_text["relations_text"]
        user = f"""
Task: Extract all SPO triples that are directly supported by the text.

Requirements:
- If applicable, please use ontology relation names.
- When using an ontology relation, respect its domain and range: the subject should match the domain type and the object should match the range type shown in the ontology.
- Also include clearly supported relations that are NOT in the ontology, using short lowerCamelCase names.
- Do NOT invent entities or facts not clearly supported by the text.
- Your final answer MUST be valid JSON only, with this schema:


{{
  "triples": [
    {{
      "subject": "string",
      "relation": "string",
      "object": "string"
    }}
  ]
}}

Ontology concepts:
{concepts_text}

Ontology relations (domain → range):
{relations_text}

Text:
"{sentence}"
"""
        return user.strip()

    def load_llama3_tokenizer(model_name: str = MODEL_NAME):
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return tok

    tokenizer = load_llama3_tokenizer()

    def make_target_json(triples: List[Dict[str, str]]) -> str:
        data = {"triples": triples}
        return json.dumps(data, ensure_ascii=False)

    def examples_from_dbpedia_domain(
        train_path: str,
        ontology_path: str,
        max_examples: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        ont = load_ontology(ontology_path)
        ont_text = ontology_to_text_dbpedia(ont)
        examples: List[Dict[str, str]] = []
        with open(train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                record = json.loads(line)
                sent = record["sent"]

                triple_dicts = []
                for t in record.get("triples", []):
                    sub = clean_label(t["sub"])
                    obj = clean_label(t["obj"])
                    rel = t.get("rel")
                    if not (sub and obj and rel):
                        continue
                    triple_dicts.append(
                        {
                            "subject": sub,
                            "relation": rel,
                            "object": obj,
                        }
                    )
                if not triple_dicts:
                    continue

                user_text = build_prompt4_user_text(ont_text, sent)
                target_json = make_target_json(triple_dicts)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_P4},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": target_json},
                ]

                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                examples.append({"text": chat_text})
        return examples

    def examples_from_wikidata_domain(
        train_path: str,
        ontology_path: str,
        max_examples: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        ont = load_ontology(ontology_path)
        ont_text = ontology_to_text_wikidata(ont)

        examples: List[Dict[str, str]] = []
        with open(train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples is not None and i >= max_examples:
                    break
                record = json.loads(line)
                sent = record["sent"]

                triple_dicts: List[Dict[str, str]] = []

                if "triples" in record:
                    for t in record.get("triples", []):
                        sub = clean_label(t.get("sub"))
                        obj = clean_label(t.get("obj"))
                        rel = t.get("rel")
                        if not (sub and obj and rel):
                            continue
                        triple_dicts.append(
                            {
                                "subject": sub,
                                "relation": rel,
                                "object": obj,
                            }
                        )
                else:
                    sub = clean_label(record.get("sub_label"))
                    obj = clean_label(record.get("obj_label"))
                    rel = record.get("rel_label")
                    if sub and obj and rel:
                        triple_dicts.append(
                            {
                                "subject": sub,
                                "relation": rel,
                                "object": obj,
                            }
                        )

                if not triple_dicts:
                    continue

                user_text = build_prompt4_user_text(ont_text, sent)
                target_json = make_target_json(triple_dicts)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_P4},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": target_json},
                ]

                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                examples.append({"text": chat_text})
        return examples

    all_dbp_examples: List[Dict[str, str]] = []
    for ont_name in DBPEDIA_ONTOLOGIES:
        id_part = ont_name.replace("ont_", "")
        train_path = os.path.join(DBPEDIA_TRAIN_DIR, f"{ont_name}_train.jsonl")
        ontology_path = os.path.join(DBPEDIA_ONTOLOGY_DIR, f"{id_part}_ontology.json")

        print(f"[INFO] Loading DBpedia domain: {ont_name}")
        domain_examples = examples_from_dbpedia_domain(
            train_path=train_path,
            ontology_path=ontology_path,
            max_examples=MAX_EXAMPLES,
        )
        print(f"       -> {len(domain_examples)} examples")
        all_dbp_examples.extend(domain_examples)

    all_wik_examples: List[Dict[str, str]] = []
    for ont_name in WIKIDATA_ONTOLOGIES:
        id_part = ont_name.replace("ont_", "")
        train_path = os.path.join(WIKIDATA_TRAIN_DIR, f"{ont_name}_train.jsonl")
        ontology_path = os.path.join(WIKIDATA_ONTOLOGY_DIR, f"{id_part}_ontology.json")

        print(f"[INFO] Loading Wikidata domain: {ont_name}")
        domain_examples = examples_from_wikidata_domain(
            train_path=train_path,
            ontology_path=ontology_path,
            max_examples=MAX_EXAMPLES,
        )
        print(f"       -> {len(domain_examples)} examples")
        all_wik_examples.extend(domain_examples)

    print(f"\n[INFO] TOTAL DBpedia examples: {len(all_dbp_examples)}")
    print(f"[INFO] TOTAL Wikidata examples: {len(all_wik_examples)}")

    dataset = concatenate_datasets(
        [
            Dataset.from_list(all_dbp_examples),
            Dataset.from_list(all_wik_examples),
        ]
    )

    split = dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"[INFO] Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    if VERBOSE:
        print("\n[DEBUG] Example training record:")
        print(train_ds[0]["text"][:10000])

        print("\n==== PRINTING ALL TRAINING EXAMPLES TO VERIFY (first 300 chars each) ====\n")
        for i in range(min(len(train_ds), 10)):
            print(f"--- Example {i} ---")
            print(train_ds[i]["text"][:30000])
            print("\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("[INFO] Model loaded with 4-bit quantization.")
    print("[INFO] Model dtype:", model.dtype)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=300,
        eval_strategy="steps",
        eval_steps=300,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to=None,
        max_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Training finished. Evaluating best model on eval set...")
    final_metrics = trainer.evaluate()
    print("[INFO] Final eval metrics:", final_metrics)

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Finished. Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_prompt4_qlora_finetune()