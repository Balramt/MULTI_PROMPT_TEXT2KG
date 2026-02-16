import os
import re
import json
import time
from textwrap import dedent
from typing import Dict, Any, List, Tuple, Optional, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub.utils import disable_progress_bars
from peft import PeftModel

disable_progress_bars()

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

ADAPTER_DIR = (
    "/upb/users/b/balram/profiles/unix/cs/promptKG/src_finetune/"
    "upb/users/b/balram/profiles/unix/cs/promptKG/finetune_models/"
    "llama3_prompt4_finetuned_ontologyAware"
)
ADAPTER_NAME = "default"

USE_4BIT = True

MAX_DEPTH = 3
ROOT_K = 2
DFS_VALUE_THRESHOLD = 0.7

VERBOSE = False
DEBUG_SHOW_FIRST_N_RECORDS = 2
DEBUG_SHOW_PROMPTS = False
DEBUG_PROMPT_MAX_CHARS = 8000
DEBUG_RAW_MAX_CHARS = 8000

WIKIDATA_INPUT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/wikidata/"
WIKIDATA_ONTO_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/wikidata"
WIKIDATA_FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/wikidata"
WIKIDATA_OUT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/TOT_dfs/wikidata/"

DBPEDIA_INPUT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/input_text/dbpedia/"
DBPEDIA_ONTO_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/dbpedia/"
DBPEDIA_FEWSHOT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/fewshots_example/dbpedia/"
DBPEDIA_OUT_DIR = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/TOT_dfs/dbpedia/"

WIKIDATA_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")
DBPEDIA_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_test\.jsonl$")

model: Optional[PeftModel] = None
tokenizer: Optional[AutoTokenizer] = None

LLM_CALL_COUNTERS = {
    "G": 0,
    "V": 0,
    "SUPPORT": 0,
}


def setup_model_llama3_raw(
    model_id: str = MODEL_ID,
    adapter_dir: str = ADAPTER_DIR,
):
    adapter_dir = os.path.abspath(adapter_dir)
    print("⏳ Loading base model:", model_id)
    print("⏳ Loading LoRA adapter from:", adapter_dir)

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )
    peft_model.eval()
    peft_model.config.use_cache = True
    peft_model.config.pad_token_id = tok.pad_token_id

    try:
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter(ADAPTER_NAME)
        elif hasattr(peft_model, "set_active_adapters"):
            peft_model.set_active_adapters(ADAPTER_NAME)
    except Exception as e:
        print("[WARN] Could not explicitly activate adapter:", e)

    return peft_model, tok


def enable_adapter_for_G():
    global model
    if isinstance(model, PeftModel):
        try:
            if hasattr(model, "set_adapter"):
                model.set_adapter(ADAPTER_NAME)
            elif hasattr(model, "set_active_adapters"):
                model.set_active_adapters(ADAPTER_NAME)
        except Exception as e:
            print("[WARN] enable_adapter_for_G failed:", e)


def disable_adapter_for_base():
    global model
    if isinstance(model, PeftModel):
        try:
            if hasattr(model, "disable_adapter"):
                model.disable_adapter()
            elif hasattr(model, "set_active_adapters"):
                model.set_active_adapters([])
        except Exception as e:
            print("[WARN] disable_adapter_for_base failed:", e)


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
            lookup[str(rid)] = {
                "example_sentence": sent,
                "example_triples": triples
            }
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


def generate_chat(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    debug_tag: str = "",
):
    assert model is not None and tokenizer is not None, "Model/tokenizer not initialized."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id

    if VERBOSE and DEBUG_SHOW_PROMPTS:
        print("\n" + "=" * 80)
        print(f"[{debug_tag}] SYSTEM PROMPT (truncated):")
        print(system_prompt[:DEBUG_PROMPT_MAX_CHARS])
        print("-" * 40)
        print(f"[{debug_tag}] USER PROMPT (truncated):")
        print(user_prompt[:DEBUG_PROMPT_MAX_CHARS])
        print("=" * 80)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    gen_ids = output_ids[0][input_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if VERBOSE:
        print("\n" + "=" * 80)
        print(f"[{debug_tag}] RAW MODEL OUTPUT (truncated):")
        print(gen_text[:DEBUG_RAW_MAX_CHARS])
        print("=" * 80 + "\n")

    return gen_text


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


def triple_key(t: Dict[str, str]) -> str:
    s = t["subject"].strip()
    r = t["relation"].strip()
    o = t["object"].strip()
    return f"{s}||{r}||{o}"


def build_spo_system_for_G_root() -> str:
    return dedent("""\
    You are an ontology-aware information extraction model trained to extract subject–predicate–object (SPO) triples from text.

    Your job:
    - Read the ontology description, the worked example (if any), and the input text.
    - Propose subject–predicate–object (SPO) triples that are clearly and directly supported by the text.
    - Whenever possible, use the ontology relation names and respect their domain and range.
    - If the text expresses a relation that is not covered by the ontology, create a short, descriptive lowerCamelCase relation name (e.g., locatedIn, knownAs, partOf).
    - Do NOT invent entities or facts that are not supported by the text.

    OUTPUT CONSTRAINTS (STRICT):
    - You MUST output valid JSON only.
    - JSON schema (required):

      {
        "triples": [
          {
            "subject": "string",
            "relation": "string",
            "object": "string"
          }
        ]
      }

    - No explanations, natural language, comments, or reasoning outside the JSON object.
    """).strip()


def build_spo_user_for_G_root(
    TEXT: str,
    ONTO: Dict[str, Any],
    few_shot_example: Optional[Dict[str, Any]] = None,
) -> str:
    example_section = ""
    if few_shot_example is not None:
        example_section = "\n\n" + _fewshot_block_spo_for_G(few_shot_example)

    return dedent(f"""\
    Task: Propose SPO triples that are directly supported by the given text, using the ontology when applicable.

    You MUST propose triples that are clearly and explicitly supported by the text.
    You MAY propose more than one triple, but they must all be directly grounded in the text.

    Ontology concepts:
    {format_ontology_concepts(ONTO)}

    Ontology relations (domain → range):
    {format_ontology_relations(ONTO)}

    {example_section}

    Text:
    "{_escape_multiline(TEXT)}"

    Instructions:
    - Output only SPO triples that are directly supported by the input text.
    - Do NOT output hallucinated or inferred facts.
    - If no valid triples exist, return "triples": [].

    Return your answer strictly in the JSON format specified in the system message.
    """).strip()


def _fewshot_block_spo_for_G(few_shot_example: Dict[str, Any]) -> str:
    ex_sent = _escape_multiline(few_shot_example.get("example_sentence", ""))
    triples = few_shot_example.get("example_triples", [])

    example_json = {
        "triples": [
            {"subject": s, "relation": r, "object": o}
            for (s, r, o) in triples
        ]
    }

    return dedent(f"""\
    Few-shot example (for guidance):

    Example sentence:
    "{ex_sent}"

    Example triples (desired JSON format and ontology usage):
    {json.dumps(example_json, ensure_ascii=False, indent=2)}
    """).strip()


def build_spo_system_for_G_state() -> str:
    return dedent("""\
    You are an ontology-aware triple expansion model.

    Your job:
    - Read the input text.
    - Read the already accepted triples.
    - Propose ONE ADDITIONAL subject–predicate–object (SPO) triple that is clearly supported by the text.

    Instructions:
    - Carefully scan every triple in the Current accepted triples list and treat them as facts that have already been extracted.
    - Do NOT invent new labels, synonyms, abbreviations, or paraphrases that are not explicitly written in the text.
    - Do NOT output any triple that is already in the Current accepted triples list.
    - Do NOT output a triple if its meaning is already covered by any triple in the Current accepted triples list, even if the wording is different.
    - If the same SUBJECT and RELATION already exist in a triple, do not output another triple with the same logical or semantic meaning.
    - Try to add a triple if it introduces a new relation or a different real-world fact.
    - Prefer to look for the next distinct SPO fact expressed in the text.
    - It is allowed to output an empty "triples" list if there are no additional supported triples.

    OUTPUT CONSTRAINTS (STRICT):
    - You MUST output valid JSON only.
    - JSON schema (required):

      {
        "triples": [
          {
            "subject": "string",
            "relation": "string",
            "object": "string"
          }
        ]
      }

    - No explanations, natural language, comments, or reasoning outside the JSON object.
    """).strip()


def build_spo_user_for_G_state(
    TEXT: str,
    ONTO: Dict[str, Any],
    current_triples: List[Dict[str, str]],
    few_shot_example: Optional[Dict[str, Any]] = None,
) -> str:
    example_section = ""
    if few_shot_example is not None:
        example_section = "\n\n" + _fewshot_block_spo_for_G(few_shot_example)

    current_json = {"triples": current_triples}

    return dedent(f"""\
    Ontology concepts:
    {format_ontology_concepts(ONTO)}

    Ontology relations (domain → range):
    {format_ontology_relations(ONTO)}

    {example_section}

    Text:
    "{_escape_multiline(TEXT)}"

    Current accepted triples list:
    {json.dumps(current_json, ensure_ascii=False, indent=2)}

    Output:
    - Valid JSON only (following the OUTPUT CONSTRAINTS defined in the system message).
    """).strip()


def build_system_for_V() -> str:
    return dedent("""\
    You are a state evaluator (V) for ontology-aware triple extraction.

    Your job:
    - Given a text, an ontology, and a set of SPO triples (a candidate state),
      evaluate how well this state is supported by the text and consistent with the ontology.

    You must:
    - Output a single scalar "stateScore" in [0,1] summarizing the overall quality of the entire set:
      - 0.9–1.0: all triples are clearly and precisely supported by the text and ontology-consistent.
      - 0.6–0.8: mostly supported; minor ambiguity, paraphrasing, or minor ontology mismatch.
      - 0.3–0.5: weakly supported; several issues, vague support, or partial hallucination.
      - 0.0–0.2: mostly unsupported, wrong, or contradictory.

    Task:
    - Evaluate how well the Candidate triple set (state) is supported by the text and consistent with the ontology.
    - Return JSON exactly in the format defined below:
      - "stateScore" in [0,1] for the entire set.

    OUTPUT CONSTRAINTS (STRICT):
    - You MUST output valid JSON only.
    - JSON schema (required):

      {
        "stateScore": 0.0
      }

    - No explanations, natural language, or comments outside the JSON object.
    """).strip()


def build_user_for_V(
    TEXT: str,
    ONTO: Dict[str, Any],
    state_triples: List[Dict[str, str]],
) -> str:
    triples_json = {"triples": state_triples}
    return dedent(f"""\
    Text:
    \"\"\"{_escape_multiline(TEXT)}\"\"\"

    Ontology concepts:
    {format_ontology_concepts(ONTO)}

    Ontology relations (domain → range):
    {format_ontology_relations(ONTO)}

    Candidate triple set (state):
    {json.dumps(triples_json, ensure_ascii=False, indent=2)}

    Output:
    - Valid JSON only (following the OUTPUT CONSTRAINTS defined in the system message).
    """).strip()


def build_system_for_support() -> str:
    return dedent("""\
    You are a support extractor for ontology-aware triples.

    Your job:
    - Carefully scan every triple in the 'Triples to support' list and treat each triple individually.
    - For each triple, return the exact supporting evidence from the text.
    - You can assume that every triple provided is correct and grounded in the text.
    - The support text MUST be a verbatim substring of the input text (case-insensitive).
    - Do NOT paraphrase, summarize, or infer missing context.
    - Do NOT output reasoning or explanation.
    - If multiple spans in the text support a triple, choose the clearest and most explicit one.

    VERY IMPORTANT:
    - For each triple, you MUST output both:
      - "triple": ["subject","relation","object"]
      - "support": "<substring from the text>"
    - Do NOT use keys named "subject", "relation", or "object" at the top level of the triple object.
      Always wrap them inside the "triple" list.

    OUTPUT CONSTRAINTS (STRICT):
    - You MUST output valid JSON only.
    - JSON schema (required):

      {
        "triples": [
          {
            "triple": ["subject","relation","object"],
            "support": "string"
          }
        ]
      }

    - No explanations or natural language outside the JSON object.
    """).strip()


def build_user_for_support(
    TEXT: str,
    ONTO: Dict[str, Any],
    state_triples: List[Dict[str, str]],
) -> str:
    triples_json = {"triples": state_triples}
    return dedent(f"""\
    Text:
    \"\"\"{_escape_multiline(TEXT)}\"\"\"

    Triples to support:
    {json.dumps(triples_json, ensure_ascii=False, indent=2)}

    Output:
    - Valid JSON only (following the OUTPUT CONSTRAINTS defined in the system message).
    """).strip()


def _parse_triples_from_json(parsed: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
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
            "subject": s.strip(),
            "relation": r.strip(),
            "object": o.strip(),
        })
    return triples


def call_G(
    TEXT: str,
    ONTO: Dict[str, Any],
    current_triples: Optional[List[Dict[str, str]]] = None,
    few_shot_example: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 384,
    debug_tag: str = "G",
) -> List[Dict[str, str]]:
    global LLM_CALL_COUNTERS
    LLM_CALL_COUNTERS["G"] += 1

    enable_adapter_for_G()

    if not current_triples:
        system_prompt = build_spo_system_for_G_root()
        user_prompt = build_spo_user_for_G_root(
            TEXT,
            ONTO,
            few_shot_example=few_shot_example
        )
        tag = f"{debug_tag}-ROOT"
    else:
        system_prompt = build_spo_system_for_G_state()
        user_prompt = build_spo_user_for_G_state(
            TEXT,
            ONTO,
            current_triples=current_triples,
            few_shot_example=few_shot_example
        )
        tag = f"{debug_tag}-STATE(size={len(current_triples)})"

    raw = generate_chat(system_prompt, user_prompt, max_new_tokens=max_new_tokens, debug_tag=tag)
    parsed = try_parse_json(raw)
    triples = _parse_triples_from_json(parsed)

    if VERBOSE:
        print(f"[{tag}] Parsed {len(triples)} triples:", triples[:10])

    return triples


def call_V(
    TEXT: str,
    ONTO: Dict[str, Any],
    state_triples: List[Dict[str, str]],
    max_new_tokens: int = 64,
    debug_tag: str = "V",
) -> float:
    global LLM_CALL_COUNTERS
    LLM_CALL_COUNTERS["V"] += 1

    disable_adapter_for_base()
    system_prompt = build_system_for_V()
    user_prompt = build_user_for_V(TEXT, ONTO, state_triples)

    tag = f"{debug_tag}-STATE(size={len(state_triples)})"
    raw = generate_chat(system_prompt, user_prompt, max_new_tokens=max_new_tokens, debug_tag=tag)
    parsed = try_parse_json(raw)

    if VERBOSE:
        print(f"[{tag}] Parsed JSON:", parsed)

    if not isinstance(parsed, dict):
        return 0.0
    try:
        score = float(parsed.get("stateScore", 0.0))
    except Exception:
        score = 0.0
    return score


def call_support(
    TEXT: str,
    ONTO: Dict[str, Any],
    state_triples: List[Dict[str, str]],
    max_new_tokens: int = 512,
    debug_tag: str = "SUPPORT",
) -> Dict[str, Any]:
    global LLM_CALL_COUNTERS
    LLM_CALL_COUNTERS["SUPPORT"] += 1

    disable_adapter_for_base()
    system_prompt = build_system_for_support()
    user_prompt = build_user_for_support(TEXT, ONTO, state_triples)

    raw = generate_chat(system_prompt, user_prompt, max_new_tokens=max_new_tokens, debug_tag=debug_tag)
    parsed = try_parse_json(raw)

    if VERBOSE:
        print(f"[{debug_tag}] Parsed JSON:", parsed)

    if not isinstance(parsed, dict):
        return {"triples": []}
    if "triples" not in parsed or not isinstance(parsed["triples"], list):
        parsed["triples"] = []
    return parsed


def call_G_root_k2(
    TEXT: str,
    ONTO: Dict[str, Any],
    few_shot_example: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    triples = call_G(
        TEXT=TEXT,
        ONTO=ONTO,
        current_triples=None,
        few_shot_example=few_shot_example,
        debug_tag="G-ROOT",
    )
    seen = set()
    root_triples: List[Dict[str, str]] = []
    for t in triples:
        k = triple_key(t)
        if k in seen:
            continue
        seen.add(k)
        root_triples.append(t)
        if len(root_triples) >= ROOT_K:
            break
    if VERBOSE:
        print(f"[G-ROOT] Using {len(root_triples)} root triples (k_root={ROOT_K})")
    return root_triples


def call_G_one_triple(
    TEXT: str,
    ONTO: Dict[str, Any],
    current_triples: List[Dict[str, str]],
    few_shot_example: Optional[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    triples = call_G(
        TEXT=TEXT,
        ONTO=ONTO,
        current_triples=current_triples,
        few_shot_example=few_shot_example,
        debug_tag="G-STATE",
    )
    existing_keys = {triple_key(t) for t in current_triples}
    for t in triples:
        k = triple_key(t)
        if k not in existing_keys:
            if VERBOSE:
                print("[G-STATE] Selected new triple:", t)
            return t
    if VERBOSE:
        print("[G-STATE] No new non-duplicate triple found.")
    return None


def tot_dfs_for_sentence(
    TEXT: str,
    ontology_json: Dict[str, Any],
    few_shot_example: Optional[Dict[str, Any]] = None,
    max_depth: int = MAX_DEPTH,
    value_threshold: float = DFS_VALUE_THRESHOLD,
    debug_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, str]], float, int, int, Dict[str, int]]:
    global LLM_CALL_COUNTERS
    for k in LLM_CALL_COUNTERS:
        LLM_CALL_COUNTERS[k] = 0

    ONTO = ontology_json

    best_state_global: List[Dict[str, str]] = []
    best_score_global: float = 0.0

    candidates_by_depth: Dict[int, List[Dict[str, Any]]] = {}

    states_evaluated: int = 0

    if VERBOSE:
        print("\n" + "#" * 80)
        print(f"[ToT-DFS] START SENTENCE | id={debug_id}")
        print("[ToT-DFS] TEXT:", TEXT)
        print("#" * 80 + "\n")

    def dfs(state: List[Dict[str, str]], depth: int, parent_rec: Optional[Dict[str, Any]]):
        nonlocal best_state_global, best_score_global, candidates_by_depth, states_evaluated

        if VERBOSE:
            print("\n" + "-" * 80)
            print(f"[ToT-DFS] DFS at depth={depth} | state size={len(state)} | id={debug_id}")
            print("[ToT-DFS] Current state triples:", state)
            print("-" * 80)

        if depth >= max_depth:
            if VERBOSE:
                print(f"[ToT-DFS] Reached max_depth={max_depth}, no further expansion.")
            return

        child_states: List[List[Dict[str, str]]] = []

        if depth == 0:
            root_triples = call_G_root_k2(TEXT, ONTO, few_shot_example=few_shot_example)
            for t in root_triples:
                child_states.append([t])
        else:
            new_triple = call_G_one_triple(
                TEXT, ONTO,
                current_triples=state,
                few_shot_example=few_shot_example
            )
            if new_triple is not None:
                child_states.append(state + [new_triple])

        if not child_states:
            if VERBOSE:
                print("[ToT-DFS] No children generated from this state. Backtracking.")
            return

        scored_children: List[Tuple[List[Dict[str, str]], float, Dict[str, Any]]] = []

        for child_state in child_states:
            score = call_V(TEXT, ONTO, child_state)
            states_evaluated += 1

            depth_child = len(child_state)

            if VERBOSE:
                print(f"[ToT-DFS] Evaluated child state (depth={depth_child}, size={len(child_state)}), "
                      f"score={score:.3f}")
                print("[ToT-DFS] Child triples:", child_state)

            if score > best_score_global:
                best_score_global = score
                best_state_global = list(child_state)
                if VERBOSE:
                    print(f"[ToT-DFS] New GLOBAL BEST (any depth) score={best_score_global:.3f} "
                          f"for state size={len(best_state_global)}")

            if score < value_threshold:
                if VERBOSE:
                    print(f"[ToT-DFS] PRUNED child: score={score:.3f} < threshold={value_threshold:.3f}")
                continue

            state_rec = {
                "triples": list(child_state),
                "score": float(score),
                "depth": int(depth_child),
                "parent": parent_rec,
            }
            candidates_by_depth.setdefault(depth_child, []).append(state_rec)

            if VERBOSE:
                print(f"[ToT-DFS] Accepted child state at depth={depth_child}, score={score:.3f}")

            scored_children.append((child_state, score, state_rec))

        if not scored_children:
            if VERBOSE:
                print("[ToT-DFS] All children pruned. Backtracking.")
            return

        scored_children.sort(key=lambda x: x[1], reverse=True)

        for child_state, score, state_rec in scored_children:
            if VERBOSE:
                print(f"[ToT-DFS] RECURSE into child (depth={len(child_state)}, size={len(child_state)}) "
                      f"with score={score:.3f}")
            dfs(child_state, depth + 1, state_rec)

    dfs(state=[], depth=0, parent_rec=None)

    if VERBOSE:
        print("\n" + "#" * 80)
        print(f"[ToT-DFS] FINISHED DFS | id={debug_id}")
        print("[ToT-DFS] GLOBAL BEST (any depth):", best_state_global)
        print(f"[ToT-DFS] GLOBAL BEST SCORE (any depth): {best_score_global:.3f}")
        print(f"[ToT-DFS] STATES EVALUATED (V calls): {states_evaluated}")
        if candidates_by_depth:
            debug_map = {}
            for d, recs in candidates_by_depth.items():
                best_rec = max(recs, key=lambda r: r["score"])
                debug_map[d] = (round(best_rec["score"], 3), f"{len(best_rec['triples'])} triples")
            print("[ToT-DFS] CANDIDATES-BY-DEPTH (best score per depth):", debug_map)
        else:
            print("[ToT-DFS] No candidates_by_depth (all states pruned).")
        print("#" * 80 + "\n")

    def build_score_path(rec: Dict[str, Any]) -> List[float]:
        path: List[float] = []
        cur = rec
        while cur is not None and cur.get("depth", 0) > 0:
            path.append(cur["score"])
            cur = cur.get("parent")
        path.reverse()
        return path

    if candidates_by_depth:
        deepest_depth = max(candidates_by_depth.keys())
        depth_candidates = candidates_by_depth[deepest_depth]

        best_rec = depth_candidates[0]
        best_path = build_score_path(best_rec)

        for rec in depth_candidates[1:]:
            path = build_score_path(rec)

            max_len = max(len(best_path), len(path))
            bp = best_path + [-1.0] * (max_len - len(best_path))
            p = path + [-1.0] * (max_len - len(path))

            better = False
            worse = False

            for i in range(max_len - 1, -1, -1):
                if p[i] > bp[i]:
                    better = True
                    break
                if p[i] < bp[i]:
                    worse = True
                    break

            if better and not worse:
                best_rec = rec
                best_path = path

        best_state = best_rec["triples"]
        best_score = best_rec["score"]

        if VERBOSE:
            print("\n" + "#" * 80)
            print(f"[ToT-DFS] FINAL CHOICE BY MAX-DEPTH + LEXICOGRAPHIC RULE | id={debug_id}")
            print(f"[ToT-DFS] Chosen depth = {deepest_depth}")
            print(f"[ToT-DFS] Chosen best_score = {best_score:.3f}")
            print(f"[ToT-DFS] Chosen score path = {best_path}")
            print("[ToT-DFS] Chosen best_state triples:", best_state)
            print("#" * 80 + "\n")
    else:
        best_state = []
        best_score = 0.0
        if VERBOSE:
            print("[ToT-DFS] No non-pruned states found; best_state is empty.")

    support_out = call_support(TEXT, ONTO, best_state, debug_tag="SUPPORT")

    triple_support_map: Dict[str, str] = {}

    for item in support_out.get("triples", []):
        tri = item.get("triple")
        if isinstance(tri, list) and len(tri) == 3:
            s, r, o = tri
        else:
            s = item.get("subject")
            r = item.get("relation")
            o = item.get("object")

        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue

        key = f"{s.strip()}||{r.strip()}||{o.strip()}"
        triple_support_map[key] = str(item.get("support", "")).strip()

    final_triples_out: List[Dict[str, Any]] = []
    for t in best_state:
        key = triple_key(t)
        support = triple_support_map.get(key, "")
        final_triples_out.append({
            "triple": [t["subject"], t["relation"], t["object"]],
            "confidence": best_score,
            "support": support,
        })

    final_output = {
        "triples": final_triples_out
    }

    llm_calls_breakdown = dict(LLM_CALL_COUNTERS)
    llm_calls_total = sum(LLM_CALL_COUNTERS.values())

    if VERBOSE:
        print("[ToT-DFS] LLM CALLS BREAKDOWN:", llm_calls_breakdown)
        print("[ToT-DFS] LLM CALLS TOTAL:", llm_calls_total)

    return final_output, best_state, best_score, states_evaluated, llm_calls_total, llm_calls_breakdown


def make_wikidata_paths(filename: str):
    m = WIKIDATA_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Unexpected Wikidata filename: {filename}")
    idx, cat = m.groups()
    input_jsonl_path = os.path.join(WIKIDATA_INPUT_DIR, filename)
    ontology_json_path = os.path.join(WIKIDATA_ONTO_DIR, f"{idx}_{cat}_ontology.json")
    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(WIKIDATA_OUT_DIR, out_name)
    few_shot_path = os.path.join(WIKIDATA_FEWSHOT_DIR, f"ont_{idx}_{cat}_few_shot.jsonl")
    if not os.path.exists(few_shot_path):
        few_shot_path = None
    return input_jsonl_path, ontology_json_path, output_jsonl_path, few_shot_path, f"ont_{idx}_{cat}"


def make_dbpedia_paths(filename: str):
    m = DBPEDIA_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Unexpected DBpedia filename: {filename}")
    idx, cat = m.groups()
    input_jsonl_path = os.path.join(DBPEDIA_INPUT_DIR, filename)
    ontology_json_path = os.path.join(DBPEDIA_ONTO_DIR, f"{idx}_{cat}_ontology.json")
    out_name = filename.replace("_test.jsonl", "_output.jsonl")
    output_jsonl_path = os.path.join(DBPEDIA_OUT_DIR, out_name)
    few_shot_path = os.path.join(DBPEDIA_FEWSHOT_DIR, f"ont_{idx}_{cat}_few_shot.jsonl")
    if not os.path.exists(few_shot_path):
        few_shot_path = None
    return input_jsonl_path, ontology_json_path, output_jsonl_path, few_shot_path, f"ont_{idx}_{cat}"


def run_tot_on_file_dfs(
    input_jsonl_path: str,
    ontology_json_path: str,
    output_jsonl_path: str,
    few_shot_path: Optional[str] = None,
    max_items: Optional[int] = None,
    verbose: bool = True,
):
    if verbose:
        print(f"[ToT-DFS RUN] INPUT={input_jsonl_path}")
        print(f"[ToT-DFS RUN] ONTO={ontology_json_path}")
        print(f"[ToT-DFS RUN] OUT ={output_jsonl_path}")
        if few_shot_path:
            print(f"[ToT-DFS RUN] FEWSHOT={few_shot_path}")
        else:
            print("[ToT-DFS RUN] FEWSHOT=None")

    ONTO = load_ontology_json(ontology_json_path)
    few_shot_lookup = load_few_shot_lookup(few_shot_path) if few_shot_path else {}

    records: List[Dict[str, Any]] = list(read_jsonl(input_jsonl_path, max_items=max_items))
    out_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        rec_id = str(rec.get("id") or f"item_{idx}")
        text_val, text_key = extract_text_field(rec)
        one_shot = pick_few_shot_for_record(rec, few_shot_lookup, use_global_fallback=False)

        if verbose and idx < DEBUG_SHOW_FIRST_N_RECORDS:
            print("\n" + "=" * 80)
            print(f"[ID] {rec_id}")
            print(f"[TEXT_KEY] {text_key}")
            print("[TEXT]", text_val)
            print(f"[HAS FEW-SHOT] {'YES' if one_shot is not None else 'NO'}")
            if one_shot is not None:
                print("[EXAMPLE SENTENCE]", one_shot.get("example_sentence", "")[:200])
                print("[EXAMPLE TRIPLES]", one_shot.get("example_triples", []))
            print("=" * 80 + "\n")

        t0 = time.time()
        final_output, best_state, best_score, states_visited, llm_calls_total, llm_calls_breakdown = tot_dfs_for_sentence(
            TEXT=text_val,
            ontology_json=ONTO,
            few_shot_example=one_shot,
            max_depth=MAX_DEPTH,
            value_threshold=DFS_VALUE_THRESHOLD,
            debug_id=rec_id,
        )
        t1 = time.time()

        if verbose and idx < DEBUG_SHOW_FIRST_N_RECORDS:
            print("\n" + "=" * 80)
            print(f"[ToT-DFS DEBUG] id={rec_id}")
            print(f"  states_visited(V calls)={states_visited}")
            print(f"  best_score={best_score:.3f}")
            print("  best_state triples:", best_state)
            print("  final_output JSON (truncated):", json.dumps(final_output, ensure_ascii=False)[:2000])
            print("  llm_calls_total:", llm_calls_total)
            print("  llm_calls_breakdown:", llm_calls_breakdown)
            print("  elapsed_sec:", t1 - t0)
            print("=" * 80 + "\n")

        out_records.append({
            "id": rec_id,
            "input_text": text_val,
            "triples": final_output.get("triples", []),
        })

    write_jsonl(output_jsonl_path, out_records)
    if verbose:
        print(f"[ToT-DFS RUN] Wrote {len(out_records)} rows -> {output_jsonl_path}")


def run_wikidata_tot_dfs():
    os.makedirs(WIKIDATA_OUT_DIR, exist_ok=True)

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

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[ToT-DFS WIKIDATA] File: {fname}")
            inp, onto, outp, fs, tag = make_wikidata_paths(fname)
            run_tot_on_file_dfs(
                input_jsonl_path=inp,
                ontology_json_path=onto,
                output_jsonl_path=outp,
                few_shot_path=fs,
                max_items=None,
                verbose=VERBOSE,
            )
            print(f"[ToT-DFS WIKIDATA] Done: {tag}")
        except Exception as exc:
            print(f"[ToT-DFS WIKIDATA ERROR] {fname}: {exc}")


def run_dbpedia_tot_dfs():
    os.makedirs(DBPEDIA_OUT_DIR, exist_ok=True)
    
    FILENAMES = [
        "ont_12_monument_test.jsonl",
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
        "ont_1_university_test.jsonl",
        "ont_2_musicalwork_test.jsonl",
        "ont_3_airport_test.jsonl",
        "ont_4_building_test.jsonl",
        "ont_5_athlete_test.jsonl",
        "ont_6_politician_test.jsonl",
        "ont_7_company_test.jsonl",
    ]

    for fname in FILENAMES:
        try:
            print("\n" + "=" * 80)
            print(f"[ToT-DFS DBPEDIA] File: {fname}")
            inp, onto, outp, fs, tag = make_dbpedia_paths(fname)
            run_tot_on_file_dfs(
                input_jsonl_path=inp,
                ontology_json_path=onto,
                output_jsonl_path=outp,
                few_shot_path=fs,
                max_items=None,
                verbose=VERBOSE,
            )
            print(f"[ToT-DFS DBPEDIA] Done: {tag}")
        except Exception as exc:
            print(f"[ToT-DFS DBPEDIA ERROR] {fname}: {exc}")


def run_all_tot_dfs():
    print("========== [ToT-DFS] SETUP MODEL ==========")
    global model, tokenizer
    model, tokenizer = setup_model_llama3_raw(MODEL_ID, ADAPTER_DIR)

    print("[DEBUG] Model class:", type(model))

    print("\n========== [ToT-DFS] RUN WIKIDATA =========")
    run_wikidata_tot_dfs()

    print("\n========== [ToT-DFS] RUN DBPEDIA ==========")
    run_dbpedia_tot_dfs()

    print("\n========== [ToT-DFS] DONE ALL DATASETS ====")


if __name__ == "__main__":
    run_all_tot_dfs()