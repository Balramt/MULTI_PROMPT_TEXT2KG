import os
import re
import json
import time
from textwrap import dedent
from typing import Dict, Any, List, Tuple, Optional, Iterable

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "../upb/users/b/balram/profiles/unix/cs/promptKG/finetune_models/llama3_prompt4_finetuned_ontologyAware"

BATCH_SIZE = 6
ADAPT_FACTOR = 0.4
ADAPT_CAP = 1000


def setup_model_llama3(model_id: str = MODEL_ID, adapter_dir: str = ADAPTER_DIR):
    adapter_dir = os.path.abspath(adapter_dir)
    print("⏳ Loading base model:", model_id)
    print("⏳ Loading LoRA adapter from:", adapter_dir)
    torch.backends.cudnn.benchmark = True

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

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
            lines.append(f"- {rel_label}({dom_label}, {rng_label})")
    return "\n".join(lines)


def build_spo_system() -> str:
    return dedent("""\
    You are an expert information extraction model for knowledge graph construction.

    Your task:
    - Read the ontology, the worked example (if present), and the new text.
    - Extract all subject-predicate-object (SPO) triples that are clearly expressed in the NEW TEXT.

    Ontology:
    - You are given ontology concepts and relations with domain and range types.
    - Whenever possible, you MUST use one of the ontology relations.
    - If the text contains a relation that is NOT covered by the ontology, you MUST still extract it:
      • Create a new predicate name that is short, meaningful, and in lowerCamelCase (e.g., locatedIn, partOf, operatedBy).
      • Respect the natural subject and object roles implied by the text.

    Output format rules:
    - You MUST output valid JSON only, and nothing else.
    - The JSON schema MUST be:
      {
        "triples": [
          {
            "subject": "string",
            "relation": "string",
            "object": "string"
          },
          ...
        ]
      }
    - The top-level object MUST have a "triples" key whose value is a list (possibly empty).
    - Do NOT include any other keys at the top level.
    - Do NOT include comments, explanations, natural language, or any extra text outside the JSON.

    Entity rules:
    - Use entity surface forms consistent with the text (e.g. "Lugenia Burns Hope", "African Americans").
    - Do NOT invent entities that are not mentioned or unambiguously implied in the text.
    - You may normalize obvious country names (e.g. "US" → "United States") when it clearly improves clarity.
    """).strip()


def _fewshot_block_spo(few_shot_example: Dict[str, Any]) -> str:
    ex_sent = _escape_multiline(few_shot_example.get("example_sentence", ""))
    triples = few_shot_example.get("example_triples", [])

    example_json = {
        "triples": [
            {
                "subject": s,
                "relation": r,
                "object": o
            }
            for (s, r, o) in triples
        ]
    }

    return dedent(f"""\
    Example sentence
    "{ex_sent}"

    Example triples (bare triples just to show *what* facts should emerge)
    {json.dumps(example_json, ensure_ascii=False, indent=2)}

    NOTE:
    - The example triples above illustrate the exact JSON format you MUST follow.
    - Your output MUST be valid JSON with the same schema:
    - Do not copy the example text; extract from the NEW TEXT section only.
    """).strip()


def build_spo_user(
    TEXT: str,
    ONTO: Dict[str, Any],
    few_shot_example: Optional[Dict[str, Any]] = None,
) -> str:
    example_section = ""
    if few_shot_example is not None:
        example_section = "\n\n" + _fewshot_block_spo(few_shot_example)

    return dedent(f"""\
    Task: Extract all SPO triples that are directly supported by the text.

    Requirements:
    - Each triple must be represented as a JSON object with keys: "subject", "relation", "object".
    - If applicable, please use ontology relation names.
    - When using an ontology relation, respect its domain and range: the subject should match the domain type and the object should match the range type shown in the ontology.
    - Also include clearly supported relations that are NOT in the ontology, using short lowerCamelCase names.
    - Do NOT invent entities or facts not clearly supported by the text.
    - Your final answer MUST be valid JSON only, with the schema:
      {{
        "triples": [
          {{
            "subject": "string",
            "relation": "string",
            "object": "string"
          }},
          ...
        ]
      }}

    Ontology concepts:
    {format_ontology_concepts(ONTO)}

    Ontology relations (domain → range):
    {format_ontology_relations(ONTO)}
    
    - The example below follows the same JSON schema as described in the Requirements above.
    {example_section}

    NEW TEXT:
    "{_escape_multiline(TEXT)}"

    Output:
    - Valid JSON only, nothing else.
    - Top-level key: "triples".
    - Each triple object MUST have "subject", "relation", "object".
    """).strip()


def generate_triple_text_batch(
    generator,
    tokenizer,
    system_texts: List[str],
    user_texts: List[str],
    dyn_max_list: List[int],
) -> List[str]:
    messages_list = []
    for sys_t, usr_t in zip(system_texts, user_texts):
        messages = [
            {"role": "system", "content": sys_t},
            {"role": "user", "content": usr_t},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        messages_list.append(formatted)

    dyn_max = int(max(dyn_max_list)) if dyn_max_list else 256

    out_batch = generator(
        messages_list,
        max_new_tokens=dyn_max,
        temperature=0.0,
        top_p=0.9,
        do_sample=False,
        return_full_text=False,
        truncation=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    results: List[str] = []
    for out in out_batch:
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            raw = out[0]["generated_text"].strip()
        elif isinstance(out, dict) and "generated_text" in out:
            raw = out["generated_text"].strip()
        else:
            raw = str(out).strip()
        results.append(raw)
    return results


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


def extract_triples_from_parsed(parsed: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    if not parsed or "triples" not in parsed:
        return triples

    for item in parsed.get("triples", []):
        if not isinstance(item, dict):
            continue
        s = item.get("subject")
        r = item.get("relation")
        o = item.get("object")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue
        triples.append({
            "sub": s.strip(),
            "rel": r.strip(),
            "obj": o.strip()
        })
    return triples


def run_pipeline_llama3_spo_json(
    ontology_path: str,
    input_jsonl_path: str,
    output_jsonl_path: str,
    max_items: Optional[int] = None,
    verbose: bool = True,
    model_id: str = MODEL_ID,
    generator=None,
    tokenizer=None,
    few_shot_path: Optional[str] = None,
):
    ontology_json = load_ontology_json(ontology_path)
    few_shot_lookup = load_few_shot_lookup(few_shot_path) if few_shot_path else {}

    local_model_loaded = False
    if generator is None or tokenizer is None:
        generator, tokenizer = setup_model_llama3(model_id=model_id)
        local_model_loaded = True

    results: List[Dict[str, Any]] = []

    recs: List[Dict[str, Any]] = list(read_jsonl(input_jsonl_path, max_items=max_items))

    system_prompts: List[str] = []
    user_prompts: List[str] = []
    meta: List[Tuple[str, str, str]] = []

    for idx, rec in enumerate(recs):
        rec_id = str(rec.get("id") or f"item_{idx}")
        text_val, text_key = extract_text_field(rec)

        one_shot = pick_few_shot_for_record(
            rec,
            few_shot_lookup,
            use_global_fallback=False,
        )

        sys_prompt = build_spo_system()
        usr_prompt = build_spo_user(text_val, ontology_json, few_shot_example=one_shot)

        if verbose and idx < 3:
            print("======================================")
            print(f"[ID] {rec_id}")
            print(f"[TEXT_KEY] {text_key}")
            print(f"[HAS FEW-SHOT] {'YES' if one_shot is not None else 'NO'}")
            if one_shot is not None:
                print("[EXAMPLE SENTENCE]", one_shot.get("example_sentence", "")[:200])
                print("[EXAMPLE TRIPLES]", one_shot.get("example_triples", []))
            print("[SYSTEM PROMPT]\n", sys_prompt)
            print("[USER PROMPT]\n", usr_prompt)
            print("[SOURCE TEXT]\n", text_val)
            print("======================================\n")

        system_prompts.append(sys_prompt)
        user_prompts.append(usr_prompt)
        meta.append((rec_id, text_val, text_key))

    total = len(user_prompts)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_sys = system_prompts[start:end]
        batch_usr = user_prompts[start:end]

        lens = []
        for s, u in zip(batch_sys, batch_usr):
            formatted = tokenizer.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False,
                add_generation_prompt=True,
            )
            L = tokenizer(formatted, return_tensors="pt")["input_ids"].shape[1]
            lens.append(L)

        dyn_values = [min(ADAPT_FACTOR * L, ADAPT_CAP) for L in lens]

        batch_idx = start // BATCH_SIZE
        if verbose and batch_idx < 2:
            print("\n==== [QWEN JSON-SPO DEBUG BATCH] Batch", batch_idx, "====")
            print("Input lengths (first 10):", lens[:10], ("..." if len(lens) > 10 else ""))
            print("Dyn values   (first 10):", [int(v) for v in dyn_values[:10]], ("..." if len(dyn_values) > 10 else ""))
            print("dyn_max used:", int(max(dyn_values) if dyn_values else ADAPT_CAP))
            print("====================================\n")

        t0 = time.time()
        raw_texts = generate_triple_text_batch(
            generator,
            tokenizer,
            batch_sys,
            batch_usr,
            dyn_values,
        )
        t1 = time.time()
        print(f"[QWEN JSON-SPO BATCH] items {start}-{end-1} / {total} | time={t1 - t0:.2f}s")

        for i, raw_response in enumerate(raw_texts):
            meta_idx = start + i
            rec_id, text_val, text_key = meta[meta_idx]

            parsed_json = try_parse_json(raw_response)
            triples = extract_triples_from_parsed(parsed_json)

            if verbose and meta_idx < 3:
                print("[RAW RESPONSE]\n", raw_response)
                print("[PARSED JSON]\n", parsed_json)
                print("[PARSED TRIPLES]\n", triples)

            out_record = {
                "id": rec_id,
                "input_text": text_val,
                "text_key": text_key,
                "triples": triples,
                "raw_output": raw_response,
            }
            results.append(out_record)

    write_jsonl(output_jsonl_path, results)

    if verbose:
        print(f"\n[QWEN JSON-SPO WRITE] {len(results)} rows -> {output_jsonl_path}")

    if local_model_loaded:
        torch.cuda.empty_cache()


WIKIDATA_PATTERN_LLAMA3 = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")

MAX_ITEMS = None
VERBOSE = True


def make_wikidata_paths_llama3(filename: str, base_input: str, base_onto: str, base_out: str):
    m = WIKIDATA_PATTERN_LLAMA3.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    idx, cat = m.groups()

    input_jsonl_path = os.path.join(base_input, filename)
    ontology_json_path = os.path.join(base_onto, f"{idx}_{cat}_ontology.json")

    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(base_out, out_name)

    tag = f"ont_{idx}_{cat}"
    return input_jsonl_path, ontology_json_path, output_jsonl_path, tag


def resolve_wikidata_fewshot_path_llama3(filename: str, fewshot_root: str) -> Optional[str]:
    m = WIKIDATA_PATTERN_LLAMA3.match(filename)
    if not m:
        return None
    idx, cat = m.groups()
    fewshot_name = f"ont_{idx}_{cat}_few_shot.jsonl"
    path = os.path.join(fewshot_root, fewshot_name)
    return path if os.path.exists(path) else None


def run_wikidata_batch_llama3(verbose: bool = True):
    BASE_INPUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/wikidata/"
    BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/wikidata/"
    FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/wikidata/"
    BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/general_extraction_prompt/wikidata/"

    FILENAMES = [
        "ont_1_movie_test.jsonl",
        "ont_2_music_test.jsonl",
        "ont_3_sport_test.jsonl",
        "ont_4_book_test.jsonl",
        "ont_5_military_test.jsonl",
        "ont_6_computer_test.jsonl",
        "ont_7_space_test.jsonl",
        "ont_8_politics_test.jsonl",
        "ont_9_nature_test.jsonl",
        "ont_10_culture_test.jsonl",
    ]

    os.makedirs(BASE_OUT, exist_ok=True)

    generator, tokenizer = setup_model_llama3(model_id=MODEL_ID)

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[RUN QWEN JSON-SPO] wikidata {fname}")

            input_jsonl_path, ontology_json_path, output_jsonl_path, tag = make_wikidata_paths_llama3(
                filename=fname,
                base_input=BASE_INPUT,
                base_onto=BASE_ONTO,
                base_out=BASE_OUT,
            )

            print(f"[INPUT ] {input_jsonl_path}")
            print(f"[ONTO  ] {ontology_json_path}")
            print(f"[OUTPUT] {output_jsonl_path}")

            fs_path = resolve_wikidata_fewshot_path_llama3(fname, FEWSHOT_DIR)
            if fs_path:
                print(f"[FEWSHOT] Using examples from: {fs_path}")
            else:
                print(f"[FEWSHOT] None found for {fname}")

            run_pipeline_llama3_spo_json(
                ontology_path=ontology_json_path,
                input_jsonl_path=input_jsonl_path,
                output_jsonl_path=output_jsonl_path,
                max_items=MAX_ITEMS,
                verbose=verbose,
                model_id=MODEL_ID,
                generator=generator,
                tokenizer=tokenizer,
                few_shot_path=fs_path,
            )

            print(f"[DONE QWEN JSON-SPO] wikidata {tag}")

        except Exception as exc:
            print(f"[ERROR QWEN JSON-SPO] wikidata {fname}: {exc}")


DBPEDIA_PATTERN_LLAMA3 = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")


def make_dbpedia_paths_llama3(filename: str, base_input: str, base_onto: str, base_out: str):
    m = DBPEDIA_PATTERN_LLAMA3.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    idx, cat = m.groups()

    input_jsonl_path = os.path.join(base_input, filename)
    ontology_json_path = os.path.join(base_onto, f"{idx}_{cat}_ontology.json")

    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(base_out, out_name)

    tag = f"ont_{idx}_{cat}"
    return input_jsonl_path, ontology_json_path, output_jsonl_path, tag


def resolve_dbpedia_fewshot_path_llama3(filename: str, fewshot_root: str) -> Optional[str]:
    m = DBPEDIA_PATTERN_LLAMA3.match(filename)
    if not m:
        return None
    idx, cat = m.groups()
    fewshot_name = f"ont_{idx}_{cat}_few_shot.jsonl"
    path = os.path.join(fewshot_root, fewshot_name)
    return path if os.path.exists(path) else None


def run_dbpedia_batch_llama3(verbose: bool = True):
    BASE_INPUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/dbpedia/"
    BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/dbpedia/"
    FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/dbpedia/"
    BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/general_extraction_prompt/dbpedia/"

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

    generator, tokenizer = setup_model_llama3(model_id=MODEL_ID)

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[RUN QWEN JSON-SPO] dbpedia {fname}")

            input_jsonl_path, ontology_json_path, output_jsonl_path, tag = make_dbpedia_paths_llama3(
                filename=fname,
                base_input=BASE_INPUT,
                base_onto=BASE_ONTO,
                base_out=BASE_OUT,
            )

            print(f"[INPUT ] {input_jsonl_path}")
            print(f"[ONTO  ] {ontology_json_path}")
            print(f"[OUTPUT] {output_jsonl_path}")

            fs_path = resolve_dbpedia_fewshot_path_llama3(fname, FEWSHOT_DIR)
            if fs_path:
                print(f"[FEWSHOT] Using examples from: {fs_path}")
            else:
                print(f"[FEWSHOT] None found for {fname}")

            run_pipeline_llama3_spo_json(
                ontology_path=ontology_json_path,
                input_jsonl_path=input_jsonl_path,
                output_jsonl_path=output_jsonl_path,
                max_items=MAX_ITEMS,
                verbose=verbose,
                model_id=MODEL_ID,
                generator=generator,
                tokenizer=tokenizer,
                few_shot_path=fs_path,
            )

            print(f"[DONE QWEN JSON-SPO] dbpedia {tag}")

        except Exception as exc:
            print(f"[ERROR QWEN JSON-SPO] dbpedia {fname}: {exc}")


if __name__ == "__main__":
    run_dbpedia_batch_llama3(verbose=True)
    pass