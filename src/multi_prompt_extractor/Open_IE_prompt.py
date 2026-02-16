import os
import re
import json
import time
from textwrap import dedent
from typing import Dict, Any, List, Tuple, Optional, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 16
ADAPT_FACTOR = 0.5
ADAPT_CAP = 2000


def setup_model(model_id: str = MODEL_ID):
    print("⏳ Loading model:", model_id)
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = True

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    return generator, tokenizer


_TEXT_KEYS_PRIORITY = ("sent", "text", "Text", "sentence", "Sentence")


def read_jsonl(path: str, max_items: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if max_items is not None and count >= max_items:
                break


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def extract_text_field(rec: Dict[str, Any]) -> Tuple[str, str]:
    for k in _TEXT_KEYS_PRIORITY:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), k

    best_key, best_val = "", ""
    for k, v in rec.items():
        if isinstance(v, str) and len(v) > len(best_val):
            best_key, best_val = k, v
    return best_val.strip(), best_key


def _escape_multiline(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


_FEWSHOT_SENT_KEYS = (
    "example_sentence", "sentence", "sent", "text", "Text",
    "Example sentence", "Example Sentence"
)
_FEWSHOT_TRIP_KEYS = (
    "example_triples", "triples", "examples",
    "Example output", "Example Output"
)


def _pick_example_sentence(rec: Dict[str, Any]) -> Optional[str]:
    for k in _FEWSHOT_SENT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _normalize_triples(triples_raw) -> Optional[List[List[str]]]:
    out: List[List[str]] = []
    if not isinstance(triples_raw, list):
        return None
    for item in triples_raw:
        if isinstance(item, list) and len(item) >= 3:
            out.append([str(item[0]), str(item[1]), str(item[2])])
        elif isinstance(item, dict):
            s = item.get("sub")
            r = item.get("rel")
            o = item.get("obj")
            if s and r and o:
                out.append([str(s), str(r), str(o)])
    return out if out else None


def _pick_example_triples(rec: Dict[str, Any]) -> Optional[List[List[str]]]:
    for k in _FEWSHOT_TRIP_KEYS:
        v = rec.get(k)
        normalized = _normalize_triples(v)
        if normalized:
            return normalized
    return None


def load_few_shot_lookup(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not path:
        return lookup
    if not os.path.exists(path):
        print(f"[FEWSHOT] File not found: {path}")
        return lookup

    total = 0
    try:
        for rec in read_jsonl(path):
            total += 1
            rid = rec.get("id")
            sent = _pick_example_sentence(rec)
            triples = _pick_example_triples(rec)
            if rid is None or not sent or not triples:
                continue
            lookup[str(rid)] = {"example_sentence": sent, "example_triples": triples}
        print(f"[FEWSHOT] loaded {len(lookup)} / {total} examples from {path}")
        if lookup:
            print("[FEWSHOT] sample ids:", list(lookup.keys())[:5])
    except Exception as e:
        print(f"[WARN] Failed to load few-shot file {path}: {e}")
    return lookup


def pick_few_shot_for_record(
    rec: Dict[str, Any],
    few_shot_lookup: Dict[str, Dict[str, Any]],
    use_global_fallback: bool = False,
) -> Optional[Dict[str, Any]]:
    fs_id = rec.get("few_shot_id")
    if isinstance(fs_id, str) and fs_id.strip():
        fs_id = fs_id.strip()
        ex = few_shot_lookup.get(fs_id)
        if ex is not None:
            return ex

    rec_id = rec.get("id")
    if isinstance(rec_id, str) and rec_id.strip():
        rec_id = rec_id.strip()
        ex = few_shot_lookup.get(rec_id)
        if ex is not None:
            return ex

    if use_global_fallback and few_shot_lookup:
        first_id = next(iter(few_shot_lookup))
        return few_shot_lookup[first_id]

    return None


def load_ontology_json(path: str) -> Dict[str, Any]:
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
            lines.append(f"- {rel_label}({dom_label},{rng_label})")
    return "\n".join(lines)


def build_p3_system() -> str:
    return (
        "You are an open IE extractor operating under a fixed ontology. From the text, propose "
        "triples [subject, relation, object] that satisfy the ontology’s domain→range. For every "
        "triple, cite exact supporting span(s) and give a 0–1 confidence. Output JSON only."
    )


def build_p3_user(
    TEXT: str,
    ONTO: Dict[str, Any],
    k: int,
    few_shot_example: Optional[Dict[str, Any]] = None,
) -> str:
    example_block = ""
    if few_shot_example is not None:
        ex_sent = _escape_multiline(few_shot_example.get("example_sentence", ""))
        ex_triples = few_shot_example.get("example_triples", [])
        example_block = dedent(f"""
        One-shot EXAMPLE (for guidance; do not copy the text):
        Example sentence:
        "{ex_sent}"

        Example triples (bare triples just to show what facts should emerge):
        {json.dumps(ex_triples, ensure_ascii=False)}

        NOTE:
        - The example triples above are simplified bare triples.
        - YOUR OUTPUT MUST follow the full schema described below
          (with subject_type, object_type, support, and confidence in [0,1]).
        - Do not copy the example text; extract from the actual Text section only.
        """).strip()

    example_section = ("\n\n" + example_block) if example_block else ""

    return dedent(f"""\
    Task: Extract up to {k} triples that are directly supported by the text. You may paraphrase, but you must quote the evidence substrings.

    Requirements
    - Extract triples [subject, relation, object] that are explicitly supported by the text.
    - For any triple that instantiates an ontology relation (i.e., its relation label appears in the ontology),
      enforce domain→range consistency: the subject_type must match the relation’s domain concept, and
      the object_type must match the relation’s range concept.
    - You may also output factual triples that do NOT correspond to any ontology relation; in that case,
      choose the most appropriate subject_type and object_type concept labels from the ontology, but do not discard the triple.
    - Return JSON only, with this schema
        {{
          "triples": [
            {{
              "triple": ["subject","relation","object"],
              "subject_type": "Concept",
              "object_type": "Concept",
              "support": "exact quote from text",
              "confidence": 0.0
            }}
          ]
        }}
    - The confidence must be a real value between 0 and 1, estimated by how clearly the text supports the triple,
      and how well subject_type and object_type match the ontology types.

    {example_section}

    Text
    "{_escape_multiline(TEXT)}"

    Ontology concepts
    {format_ontology_concepts(ONTO)}

    Ontology relations (domain → range)
    {format_ontology_relations(ONTO)}

    Constraints
    - Extract ALL clearly stated factual triples in the text (not just those that match ontology relations).
    - Do not discard a triple solely because it does not match any ontology relation; instead,
      assign the best-fitting subject_type and object_type concepts you can infer from the ontology.
    - For triples whose relation label appears in the ontology, respect the domain and range for subject_type and object_type.
    - Always extract any explicit date, time, or year mentioned in the text as part of a factual triple when relevant.
    - Resolve pronouns to the nearest valid antecedent if needed to form a correct triple.
    - Do not invent entities that are not mentioned in the text.
    - Output MUST be valid JSON and nothing else.
    """).strip()


def generate_raw_json(
    generator,
    tokenizer,
    system_text: str,
    user_text: str,
    max_new_tokens: int = 900,
    temperature: float = 0,
) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    out = generator(
        formatted,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        do_sample=False,
        return_full_text=False,
        truncation=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0]["generated_text"] if isinstance(out[0], dict) else out[0]


def try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
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


def run_pipeline_prompt3(
    ontology_path: str,
    input_jsonl_path: str,
    output_jsonl_path: str,
    k_triples: int = 5,
    max_items: Optional[int] = None,
    verbose: bool = True,
    model_id: str = MODEL_ID,
    generator=None,
    tokenizer=None,
    few_shot_path: Optional[str] = None,
):
    ontology_json = load_ontology_json(ontology_path)
    few_shot_lookup = load_few_shot_lookup(few_shot_path) if few_shot_path else {}

    if generator is None or tokenizer is None:
        generator, tokenizer = setup_model(model_id=model_id)

    results: List[Dict[str, Any]] = []

    recs: List[Dict[str, Any]] = list(read_jsonl(input_jsonl_path, max_items=max_items))

    prompt_texts: List[str] = []
    meta: List[Tuple[str, str, str, str, str]] = []

    for idx, rec in enumerate(recs):
        rec_id = str(rec.get("id") or f"item_{idx}")
        text_val, text_key = extract_text_field(rec)

        one_shot = pick_few_shot_for_record(
            rec,
            few_shot_lookup,
            use_global_fallback=False,
        )

        sys_prompt = build_p3_system()
        usr_prompt = build_p3_user(text_val, ontology_json, k_triples, few_shot_example=one_shot)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_texts.append(prompt_text)
        meta.append((rec_id, text_val, sys_prompt, usr_prompt, text_key))

    total = len(prompt_texts)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_prompts = prompt_texts[start:end]

        lens = [tokenizer(p, return_tensors="pt")["input_ids"].shape[1] for p in batch_prompts]

        dyn_values = [min(ADAPT_FACTOR * L, ADAPT_CAP) for L in lens]
        dyn_max = max(dyn_values) if dyn_values else ADAPT_CAP

        t0 = time.time()
        out_batch = generator(
            batch_prompts,
            max_new_tokens=int(dyn_max),
            temperature=0,
            top_p=1.0,
            do_sample=False,
            return_full_text=False,
            truncation=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        t1 = time.time()
        print(f"[P3 BATCH] items {start}-{end-1} / {total} | dyn_max={int(dyn_max)} | time={t1 - t0:.2f}s")

        for i, out in enumerate(out_batch):
            meta_idx = start + i
            rec_id, text_val, sys_prompt, usr_prompt, text_key = meta[meta_idx]

            if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                raw_response = out[0]["generated_text"].strip()
            elif isinstance(out, dict) and "generated_text" in out:
                raw_response = out["generated_text"].strip()
            else:
                raw_response = str(out).strip()

            parsed_json = try_parse_json(raw_response)

            out_record = {
                "id": rec_id,
                "input text": text_val,
                "prompts": {
                    "system_prompt": sys_prompt,
                    "user_prompt": usr_prompt,
                },
                "response": {
                    "LLM_output": raw_response,
                    "json": parsed_json,
                },
            }
            results.append(out_record)

    write_jsonl(output_jsonl_path, results)

    if verbose:
        print(f"\n[P3 WRITE] {len(results)} rows -> {output_jsonl_path}")


WIKIDATA_PATTERN_P3 = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")

MAX_ITEMS = None
VERBOSE = True
K_TRIPLES_P3 = 6


def make_wikidata_paths_p3(filename: str, base_input: str, base_onto: str, base_out: str):
    m = WIKIDATA_PATTERN_P3.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    idx, cat = m.groups()

    input_jsonl_path = os.path.join(base_input, filename)
    ontology_json_path = os.path.join(base_onto, f"{idx}_{cat}_ontology.json")

    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(base_out, out_name)

    tag = f"ont_{idx}_{cat}"
    return input_jsonl_path, ontology_json_path, output_jsonl_path, tag


def resolve_wikidata_fewshot_path_p3(filename: str, fewshot_root: str) -> Optional[str]:
    m = WIKIDATA_PATTERN_P3.match(filename)
    if not m:
        return None
    idx, cat = m.groups()
    fewshot_name = f"ont_{idx}_{cat}_few_shot.jsonl"
    path = os.path.join(fewshot_root, fewshot_name)
    return path if os.path.exists(path) else None


def run_wikidata_batch_p3(verbose: bool = True):
    BASE_INPUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/wikidata/"
    BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/wikidata/"
    FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/wikidata/"
    BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/Open_IE_prompt/wikidata/"

    FILENAMES = [
        "ont_10_culture_test.jsonl",
        "ont_1_movie_test.jsonl",
        "ont_2_music_test.jsonl",
        "ont_3_sport_test.jsonl",
        "ont_4_book_test.jsonl",
        "ont_5_military_test.jsonl",
        "ont_6_computer_test.jsonl",
        "ont_7_space_test.jsonl",
        "ont_8_politics_test.jsonl",
        "ont_9_nature_test.jsonl",
    ]

    os.makedirs(BASE_OUT, exist_ok=True)

    generator, tokenizer = setup_model(model_id=MODEL_ID)

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[RUN P3] wikidata {fname}")

            input_jsonl_path, ontology_json_path, output_jsonl_path, tag = make_wikidata_paths_p3(
                filename=fname,
                base_input=BASE_INPUT,
                base_onto=BASE_ONTO,
                base_out=BASE_OUT,
            )

            print(f"[INPUT ] {input_jsonl_path}")
            print(f"[ONTO  ] {ontology_json_path}")
            print(f"[OUTPUT] {output_jsonl_path}")

            fs_path = resolve_wikidata_fewshot_path_p3(fname, FEWSHOT_DIR)
            if fs_path:
                print(f"[FEWSHOT] Using examples from: {fs_path}")
            else:
                print(f"[FEWSHOT] None found for {fname}")

            run_pipeline_prompt3(
                ontology_path=ontology_json_path,
                input_jsonl_path=input_jsonl_path,
                output_jsonl_path=output_jsonl_path,
                k_triples=K_TRIPLES_P3,
                max_items=MAX_ITEMS,
                verbose=verbose,
                model_id=MODEL_ID,
                generator=generator,
                tokenizer=tokenizer,
                few_shot_path=fs_path,
            )

            print(f"[DONE P3] wikidata {tag}")

        except Exception as exc:
            print(f"[ERROR P3] wikidata {fname}: {exc}")


DBPEDIA_PATTERN_P3 = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")

MAX_ITEMS = None
VERBOSE = True
K_TRIPLES_P3 = 6


def make_dbpedia_paths_p3(filename: str, base_input: str, base_onto: str, base_out: str):
    m = DBPEDIA_PATTERN_P3.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    idx, cat = m.groups()

    input_jsonl_path = os.path.join(base_input, filename)
    ontology_json_path = os.path.join(base_onto, f"{idx}_{cat}_ontology.json")

    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(base_out, out_name)

    tag = f"ont_{idx}_{cat}"
    return input_jsonl_path, ontology_json_path, output_jsonl_path, tag


def resolve_dbpedia_fewshot_path_p3(filename: str, fewshot_root: str) -> Optional[str]:
    m = DBPEDIA_PATTERN_P3.match(filename)
    if not m:
        return None
    idx, cat = m.groups()
    fewshot_name = f"ont_{idx}_{cat}_few_shot.jsonl"
    path = os.path.join(fewshot_root, fewshot_name)
    return path if os.path.exists(path) else None


def run_dbpedia_batch_p3(verbose: bool = True):
    BASE_INPUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/dbpedia/"
    BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/dbpedia/"
    FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/dbpedia/"
    BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/Open_IE_prompt/dbpedia/"

    FILENAMES = [
        "ont_12_monument_test.jsonl",
        "ont_1_university_test.jsonl",
        "ont_2_musicalwork_test.jsonl",
        "ont_3_airport_test.jsonl",
        "ont_4_building_test.jsonl",
        "ont_5_athlete_test.jsonl",
        "ont_6_politician_test.jsonl",
        "ont_7_company_test.jsonl",
        "ont_8_celestialbody_test.jsonl",
        "ont_9_astronaut_test.jsonl",
        "ont_10_comicscharacter_test.jsonl",
        "ont_11_meanoftransportation_test.jsonl",
        "ont_13_food_test.jsonl",
        "ont_14_writtenwork_test.jsonl",
        "ont_15_sportsteam_test.jsonl",
        "ont_16_city_test.jsonl",
        "ont_17_artist_test.jsonl",
        "ont_18_scientist_test.jsonl",
        "ont_19_film_test.jsonl",
    ]

    os.makedirs(BASE_OUT, exist_ok=True)

    generator, tokenizer = setup_model(model_id=MODEL_ID)

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[RUN P3] dbpedia {fname}")

            input_jsonl_path, ontology_json_path, output_jsonl_path, tag = make_dbpedia_paths_p3(
                filename=fname,
                base_input=BASE_INPUT,
                base_onto=BASE_ONTO,
                base_out=BASE_OUT,
            )

            print(f"[INPUT ] {input_jsonl_path}")
            print(f"[ONTO  ] {ontology_json_path}")
            print(f"[OUTPUT] {output_jsonl_path}")

            fs_path = resolve_dbpedia_fewshot_path_p3(fname, FEWSHOT_DIR)
            if fs_path:
                print(f"[FEWSHOT] Using examples from: {fs_path}")
            else:
                print(f"[FEWSHOT] None found for {fname}")

            run_pipeline_prompt3(
                ontology_path=ontology_json_path,
                input_jsonl_path=input_jsonl_path,
                output_jsonl_path=output_jsonl_path,
                k_triples=K_TRIPLES_P3,
                max_items=MAX_ITEMS,
                verbose=verbose,
                model_id=MODEL_ID,
                generator=generator,
                tokenizer=tokenizer,
                few_shot_path=fs_path,
            )

            print(f"[DONE P3] dbpedia {tag}")

        except Exception as exc:
            print(f"[ERROR P3] dbpedia {fname}: {exc}")


if __name__ == "__main__":
    run_wikidata_batch_p3()
    run_dbpedia_batch_p3()