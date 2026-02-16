import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TEACHER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEVICE_MAP = "auto"

BASE_TRAIN_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/train_data/wikidata/synthetic_train_data/wikidata_input_train/"
BASE_ONTOLOGY_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/ontology/old_ontology/wikidata/"
BASE_OUTPUT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/train_data/wikidata/synthetic_train_data/wikidata_output_train/"

WIKIDATA_TRAIN_FILENAMES = [
    "ont_3_sport_train.jsonl",
    "ont_4_book_train.jsonl",
    "ont_6_computer_train.jsonl",
    "ont_7_space_train.jsonl",
    "ont_8_politics_train.jsonl",
    "ont_9_nature_train.jsonl",
    "ont_10_culture_train.jsonl",
]

WIKIDATA_PATTERN_TEACHER = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_train\.jsonl$")
DEBUG_NUM_PROMPTS = 4


def load_teacher_model(model_id: str = TEACHER_MODEL_ID):
    print(f"[INFO] Loading teacher model: {model_id}")
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE_MAP,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.use_cache = True

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_ontology(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_concept_index(ontology_json: Dict[str, Any]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for c in ontology_json.get("concepts", []):
        label = str(c.get("label", "")).strip()
        if not label:
            continue
        for keyname in ("qid", "id", "label"):
            val = c.get(keyname)
            if val is None:
                continue
            sval = str(val).strip()
            if sval:
                idx[sval] = label
    return idx


def _label_for(value: Any, cindex: Dict[str, str]) -> str:
    if value is None:
        return ""
    sval = str(value).strip()
    return cindex.get(sval, sval)


def format_ontology_concepts(ontology_json: Dict[str, Any]) -> str:
    labels: List[str] = []
    for c in ontology_json.get("concepts", []):
        lab = str(c.get("label", "")).strip()
        if lab:
            labels.append(lab)
    return ", ".join(labels)


def format_ontology_relations(ontology_json: Dict[str, Any]) -> str:
    cindex = _build_concept_index(ontology_json)
    lines: List[str] = []
    for r in ontology_json.get("relations", []):
        rel_label = str(r.get("label", "")).strip()
        dom_label = _label_for(r.get("domain"), cindex)
        rng_label = _label_for(r.get("range"), cindex)
        if rel_label:
            lines.append(f"- {rel_label}({dom_label}, {rng_label})")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are an expert information extraction model for knowledge graph construction.\n\n"
        "You are given:\n"
        "- An ontology (concepts and relations with domain and range types).\n"
        "- A sentence.\n"
        "- A list of already-annotated SPO triples (seed triples) for that sentence.\n\n"
        "Your task:\n"
        "1) Extract all subject–predicate–object (SPO) triples that are directly supported by the sentence.\n"
        "2) You MUST include all seed triples exactly as given (same subject, relation, object).\n"
        "3) In addition to the seed triples, you SHOULD actively look for and add any other SPO triples that are clearly and explicitly expressed in the sentence (i.e., triples that are not already in the seed list).\n"
        "4) Use ontology relation names whenever they apply.\n"
        "5) If a relation is clearly supported but not covered by the ontology, you may use a short lowerCamelCase name.\n"
        "6) Do NOT invent entities or facts not clearly supported by the text.\n\n"
        "Output:\n"
        "- You MUST output valid JSON only, with this schema:\n"
        "{\n"
        "  \"triples\": [\n"
        "    {\n"
        "      \"subject\": \"string\",\n"
        "      \"relation\": \"string\",\n"
        "      \"object\": \"string\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "- No extra text, no comments, no explanations outside of the JSON.\n"
    ).strip()


def _escape_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_user_prompt(sentence: str, ontology_json: Dict[str, Any], seed_triples: List[Dict[str, str]]) -> str:
    concepts_text = format_ontology_concepts(ontology_json)
    relations_text = format_ontology_relations(ontology_json)

    seed_json = {
        "triples": [
            {
                "subject": t["subject"],
                "relation": t["relation"],
                "object": t["object"],
            }
            for t in seed_triples
        ]
    }

    return (
        "Task: Extract all SPO triples that are directly supported by the sentence.\n\n"
        "Ontology concepts:\n"
        f"{concepts_text}\n\n"
        "Ontology relations (domain → range):\n"
        f"{relations_text}\n\n"
        "Sentence:\n"
        f"\"{_escape_text(sentence)}\"\n\n"
        "Seed triples already annotated for this sentence:\n"
        f"{json.dumps(seed_json, ensure_ascii=False, indent=2)}\n\n"
        "Instructions:\n"
        "- Return all SPO triples that are directly supported by the sentence.\n"
        "- You MUST:\n"
        "  - include all seed triples exactly as they are,\n"
        "  - and you SHOULD add all other SPO triples that are clearly supported by the sentence and are not already present in the seed list.\n"
        "- Use ontology relation names when possible.\n"
        "- If you really need a relation that is not in the ontology, use a short lowerCamelCase name.\n\n"
        "Remember: respond with JSON only, using this schema:\n"
        "{\n"
        "  \"triples\": [\n"
        "    {\n"
        "      \"subject\": \"string\",\n"
        "      \"relation\": \"string\",\n"
        "      \"object\": \"string\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    ).strip()


def call_teacher(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            return None
    return None


def extract_triples(parsed: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    if not parsed or "triples" not in parsed:
        return triples
    for item in parsed.get("triples", []):
        if not isinstance(item, dict):
            continue
        s = item.get("subject")
        r = item.get("relation")
        o = item.get("object")
        if isinstance(s, str) and isinstance(r, str) and isinstance(o, str):
            triples.append(
                {
                    "subject": s.strip(),
                    "relation": r.strip(),
                    "object": o.strip(),
                }
            )
    return triples


def dedup_triples(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for t in triples:
        key = (t["subject"].lower(), t["relation"].lower(), t["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def enrich_wikidata_train_with_model(
    train_path: str,
    ontology_json: Dict[str, Any],
    output_path: str,
    model,
    tokenizer,
    max_items: Optional[int] = None,
):
    print(f"[INFO] Enriching train file: {train_path}")
    print(f"[INFO] Output ->             {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out_f = open(output_path, "w", encoding="utf-8")

    with open(train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                print(f"[INFO] Reached max_items={max_items}, stopping.")
                break

            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            rec_id = str(rec.get("id", f"item_{i}"))

            sent = rec.get("sent") or rec.get("sentence") or rec.get("text")
            if not isinstance(sent, str):
                print(f"[WARN] No sentence text for id={rec_id}, skipping.")
                continue
            sent = sent.strip()
            if not sent:
                print(f"[WARN] Empty sentence for id={rec_id}, skipping.")
                continue

            sub_label = str(rec.get("sub_label", "")).strip()
            obj_label = str(rec.get("obj_label", "")).strip()
            rel_label = str(rec.get("rel_label", "")).strip()

            if not (sub_label and obj_label and rel_label):
                print(f"[WARN] Missing triple fields for id={rec_id}, skipping.")
                continue

            seed_triple = {
                "subject": sub_label,
                "relation": rel_label,
                "object": obj_label,
            }

            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(sent, ontology_json, [seed_triple])

            if i < DEBUG_NUM_PROMPTS:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                print("\n" + "*" * 80)
                print(f"[DEBUG PROMPT] index={i}, id={rec_id}")
                print("---- SYSTEM PROMPT ----")
                print(system_prompt)
                print("---- USER PROMPT ----")
                print(user_prompt)
                print("---- FULL CHAT-TEMPLATE PROMPT (STRING FED TO MODEL) ----")
                print(full_prompt)
                print("*" * 80)

            raw_output = call_teacher(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=256,
            )

            parsed = try_parse_json(raw_output)
            predicted_triples = extract_triples(parsed)

            all_triples = [seed_triple] + predicted_triples
            all_triples = dedup_triples(all_triples)

            if i < DEBUG_NUM_PROMPTS:
                print(f"\n[DEBUG OUTPUT] index={i}, id={rec_id}")
                print("---- RAW MODEL OUTPUT (truncated) ----")
                print(raw_output[:800])
                print("---- PARSED JSON ----")
                print(parsed)
                print("---- PREDICTED TRIPLES (without seed) ----")
                print(predicted_triples)
                print("---- FINAL TOTAL_TRIPLES (seed + extras, deduped) ----")
                print(all_triples)
                print("*" * 80 + "\n")

            out_record = {
                "id": rec_id,
                "sent": sent,
                "llm_output": raw_output,
                "total_triples": all_triples,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    out_f.close()
    print("[INFO] Done. Enriched train file written to:", output_path)


def make_wikidata_paths_teacher(
    filename: str,
    base_train: str,
    base_onto: str,
    base_out: str,
) -> Tuple[str, str, str, str]:
    m = WIKIDATA_PATTERN_TEACHER.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    idx, cat = m.groups()

    train_path = os.path.join(base_train, filename)
    ontology_path = os.path.join(base_onto, f"{idx}_{cat}_ontology.json")

    out_name = filename.replace("_train.jsonl", "_train_enriched.jsonl")
    output_path = os.path.join(base_out, out_name)

    tag = f"ont_{idx}_{cat}"
    return train_path, ontology_path, output_path, tag


def run_wikidata_batch_teacher(max_items: Optional[int] = None):
    print("[INFO] Running teacher enrichment for all Wikidata train files.")
    print(f"[INFO] Base train dir:   {BASE_TRAIN_DIR}")
    print(f"[INFO] Base ontology dir:{BASE_ONTOLOGY_DIR}")
    print(f"[INFO] Base output dir:  {BASE_OUTPUT_DIR}")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    model, tokenizer = load_teacher_model()

    for fname in WIKIDATA_TRAIN_FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[RUN TEACHER ENRICH] wikidata {fname}")

            train_path, ontology_path, output_path, tag = make_wikidata_paths_teacher(
                filename=fname,
                base_train=BASE_TRAIN_DIR,
                base_onto=BASE_ONTOLOGY_DIR,
                base_out=BASE_OUTPUT_DIR,
            )

            print(f"[TRAIN ] {train_path}")
            print(f"[ONTO  ] {ontology_path}")
            print(f"[OUTPUT] {output_path}")

            ontology_json = load_ontology(ontology_path)

            enrich_wikidata_train_with_model(
                train_path=train_path,
                ontology_json=ontology_json,
                output_path=output_path,
                model=model,
                tokenizer=tokenizer,
                max_items=max_items,
            )

            print(f"[DONE TEACHER ENRICH] wikidata {tag}")

        except Exception as exc:
            print(f"[ERROR TEACHER ENRICH] wikidata {fname}: {exc}")


if __name__ == "__main__":
    run_wikidata_batch_teacher(
        max_items=None,
    )