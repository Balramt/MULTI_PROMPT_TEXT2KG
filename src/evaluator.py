from __future__ import annotations
import json
import os
import re
import difflib
from typing import Any, Dict, List, Tuple, Optional

DBPEDIA_P1_BASE = os.getenv(
    "DBPEDIA_P1_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/TOT_dfs/dbpedia/"
)

DBPEDIA_P2_BASE = os.getenv(
    "DBPEDIA_P2_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/Open_IE_prompt/dbpedia/"
)

DBPEDIA_P3_BASE = os.getenv(
    "DBPEDIA_P3_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/general_extraction_prompt/dbpedia/"
)

DBPEDIA_FINAL_BASE = os.getenv(
    "DBPEDIA_FINAL_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/evaluator_filtered_output/dbpedia1/"
)

DBPEDIA_FILENAMES = [
    "ont_1_university_output.jsonl",
    "ont_2_musicalwork_output.jsonl",
    "ont_3_airport_output.jsonl",
    "ont_4_building_output.jsonl",
    "ont_5_athlete_output.jsonl",
    "ont_6_politician_output.jsonl",
    "ont_7_company_output.jsonl",
    "ont_8_celestialbody_output.jsonl",
    "ont_9_astronaut_output.jsonl",
    "ont_10_comicscharacter_output.jsonl",
    "ont_11_meanoftransportation_output.jsonl",
    "ont_12_monument_output.jsonl",
    "ont_13_food_output.jsonl",
    "ont_14_writtenwork_output.jsonl",
    "ont_15_sportsteam_output.jsonl",
    "ont_16_city_output.jsonl",
    "ont_17_artist_output.jsonl",
    "ont_18_scientist_output.jsonl",
    "ont_19_film_output.jsonl",
]

WIKI_P1_BASE = os.getenv(
    "WIKI_P1_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/TOT_dfs/wikidata/"
)

WIKI_P2_BASE = os.getenv(
    "WIKI_P2_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/Open_IE_prompt/wikidata/"
)

WIKI_P3_BASE = os.getenv(
    "WIKI_P3_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/multi_step_prompts/general_extraction_prompt/wikidata/"
)

WIKI_FINAL_BASE = os.getenv(
    "WIKI_FINAL_BASE",
    "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/evaluator_filtered_output/wikidata1/"
)

WIKI_FILENAMES = [
    "ont_1_movie_output.jsonl",
    "ont_2_music_output.jsonl",
    "ont_3_sport_output.jsonl",
    "ont_4_book_output.jsonl",
    "ont_5_military_output.jsonl",
    "ont_6_computer_output.jsonl",
    "ont_7_space_output.jsonl",
    "ont_8_politics_output.jsonl",
    "ont_9_nature_output.jsonl",
    "ont_10_culture_output.jsonl",
]

DBPEDIA_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_output\.jsonl$")
WIKI_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_output\.jsonl$")

def extract_domain_from_filename(filename: str, kb_type: str) -> Optional[str]:
    pat = DBPEDIA_PATTERN if kb_type == "dbpedia" else WIKI_PATTERN
    m = pat.match(filename)
    if not m:
        return None
    return m.group(2)

_WHITESPACE_RE = re.compile(r"\s+")
_QUOTES_TO_STRIP = "“”\"'`’"

def norm_surface(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    t = s.strip().strip(_QUOTES_TO_STRIP)
    t = _WHITESPACE_RE.sub(" ", t)
    return t

def norm_key(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    t = s.strip().strip(_QUOTES_TO_STRIP)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.lower()

def canonical_triple_key(subj: Any, rel: Any, obj: Any) -> Tuple[str, str, str]:
    s = norm_key(subj)
    r = norm_key(norm_surface(rel))
    o = norm_key(norm_surface(obj))
    return (s, r, o)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj["_line_no"] = ln
                    rows.append(obj)
                except Exception as e:
                    rows.append({
                        "_line_no": ln,
                        "_parse_error": str(e),
                        "_raw": line[:500],
                    })
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
    return rows

def _extract_support_text(triple_obj: Dict[str, Any]) -> Optional[str]:
    sup = triple_obj.get("support")
    if sup is None:
        return None
    if isinstance(sup, list):
        quotes = []
        for item in sup:
            if isinstance(item, dict):
                q = item.get("quote")
                if isinstance(q, str) and q.strip():
                    quotes.append(q.strip())
            elif isinstance(item, str) and item.strip():
                quotes.append(item.strip())
        return " | ".join(quotes) if quotes else None
    if isinstance(sup, str):
        s = sup.strip()
        return s if s else None
    return None

def extract_triples_strict(row: Dict[str, Any],
                           file_tag: str,
                           domain: str,
                           kb_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def _add_triple(s_raw, p_raw, o_raw, support_field=None, confidence_field=None):
        if s_raw is None or p_raw is None or o_raw is None:
            return
        s = s_raw if isinstance(s_raw, str) else norm_surface(s_raw)
        p = p_raw if isinstance(p_raw, str) else norm_surface(p_raw)
        o = o_raw if isinstance(o_raw, str) else norm_surface(o_raw)
        conf = confidence_field
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None
        support_raw = support_field
        support_text = None
        if isinstance(support_field, str):
            support_text = support_field.strip() or None
        elif isinstance(support_field, list) or isinstance(support_field, dict):
            support_text = _extract_support_text({"support": support_field})
        out.append({
            "s": s,
            "p": p,
            "o": o,
            "confidence": conf,
            "support_raw": support_raw,
            "support_text": support_text,
            "source_prompt": file_tag,
            "domain": domain,
            "kb_type": kb_type,
            "canonical": canonical_triple_key(s, p, o),
        })

    resp = row.get("response")
    if isinstance(resp, dict):
        js = resp.get("json")
        if isinstance(js, dict):
            triples = js.get("triples")
            if isinstance(triples, list):
                for t in triples:
                    if not isinstance(t, dict):
                        continue
                    spo = t.get("triple")
                    if not (isinstance(spo, list) and len(spo) == 3):
                        continue
                    _add_triple(
                        spo[0], spo[1], spo[2],
                        support_field=t.get("support"),
                        confidence_field=t.get("confidence"),
                    )
    if out:
        return out

    top_triples = row.get("triples")
    if isinstance(top_triples, list):
        for t in top_triples:
            if not isinstance(t, dict):
                continue
            if "sub" in t and "rel" in t and "obj" in t:
                s_raw, p_raw, o_raw = t.get("sub"), t.get("rel"), t.get("obj")
            elif "subject" in t and "relation" in t and "object" in t:
                s_raw, p_raw, o_raw = t.get("subject"), t.get("relation"), t.get("object")
            elif "triple" in t and isinstance(t.get("triple"), list) and len(t["triple"]) == 3:
                s_raw, p_raw, o_raw = t["triple"]
            else:
                continue
            _add_triple(
                s_raw, p_raw, o_raw,
                support_field=t.get("support"),
                confidence_field=t.get("confidence"),
            )
    if out:
        return out

    raw_output = row.get("raw_output")
    if isinstance(raw_output, str):
        try:
            raw_obj = json.loads(raw_output)
            triples = raw_obj.get("triples")
            if isinstance(triples, list):
                for t in triples:
                    if not isinstance(t, dict):
                        continue
                    s_raw = t.get("subject")
                    p_raw = t.get("relation")
                    o_raw = t.get("object")
                    if s_raw is None or p_raw is None or o_raw is None:
                        continue
                    _add_triple(
                        s_raw, p_raw, o_raw,
                        support_field=t.get("support"),
                        confidence_field=t.get("confidence"),
                    )
        except Exception:
            pass
    return out

def _pick_input_text(existing: Optional[str], candidate_row: Dict[str, Any]) -> Optional[str]:
    if existing and isinstance(existing, str) and existing.strip():
        return existing
    txt = candidate_row.get("input text")
    if isinstance(txt, str) and txt.strip():
        return txt
    for k in ("input_text", "input", "text", "source_text"):
        v = candidate_row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return existing

def load_prompt_outputs_strict(path: str,
                              file_tag: str,
                              domain: str,
                              kb_type: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    rows = read_jsonl(path)
    for row in rows:
        if row.get("_parse_error"):
            continue
        rid = row.get("id")
        if not rid:
            continue
        bucket = data.setdefault(rid, {"input_text": None, "rows": [], "triples": []})
        bucket["rows"].append(row)
        bucket["input_text"] = _pick_input_text(bucket["input_text"], row)
        triples = extract_triples_strict(row, file_tag, domain, kb_type)
        if triples:
            bucket["triples"].extend(triples)
    return data

def dedup_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for t in triples:
        k = t.get("canonical")
        if not k:
            k = canonical_triple_key(t.get("s"), t.get("p"), t.get("o"))
            t["canonical"] = k
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

def build_id_index(P1: Dict[str, Dict[str, Any]],
                   P2: Dict[str, Dict[str, Any]],
                   P3: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ids = sorted(
        set(P1.keys()) | set(P2.keys()) | set(P3.keys()),
        key=lambda x: int(re.findall(r"\d+$", x)[0]) if re.findall(r"\d+$", x) else x
    )
    out: Dict[str, Dict[str, Any]] = {}
    for rid in ids:
        input_text = None
        for src in (P1.get(rid), P2.get(rid), P3.get(rid)):
            if src:
                input_text = _pick_input_text(input_text, {"input text": src.get("input_text")})
        p1_tr = dedup_triples(P1.get(rid, {}).get("triples", []))
        p2_tr = dedup_triples(P2.get(rid, {}).get("triples", []))
        p3_tr = dedup_triples(P3.get(rid, {}).get("triples", []))
        out[rid] = {
            "input_text": input_text,
            "p1": p1_tr,
            "p2": p2_tr,
            "p3": p3_tr,
        }
    return out

def summarize_loaded(index_by_id: Dict[str, Dict[str, Any]]) -> None:
    total_ids = len(index_by_id)
    p1 = sum(len(b["p1"]) for b in index_by_id.values())
    p2 = sum(len(b["p2"]) for b in index_by_id.values())
    p3 = sum(len(b["p3"]) for b in index_by_id.values())
    print(f"[Loaded] ids={total_ids} | triples: p1={p1}, p2={p2}, p3={p3} | total={p1+p2+p3}")

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def _norm_for_match(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = _WS_RE.sub(" ", t)
    return t

def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall(_norm_for_match(s))

def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

def fuzzy_in(needle: str, haystack: str, threshold: float = 0.90) -> Tuple[bool, float]:
    n = _norm_for_match(needle)
    h = _norm_for_match(haystack)
    if not n or not h:
        return (False, 0.0)
    if n in h:
        return (True, 1.0)
    n_tokens = _tokens(n)
    h_tokens = _tokens(h)
    if not n_tokens or not h_tokens:
        score = difflib.SequenceMatcher(None, h, n).ratio()
        return (score >= threshold, score)
    best = 0.0
    win_min = max(1, len(n_tokens))
    win_max = min(len(h_tokens), len(n_tokens) + 2)
    for w in range(win_min, win_max + 1):
        for i in range(0, len(h_tokens) - w + 1):
            seg = " ".join(h_tokens[i:i + w])
            r = difflib.SequenceMatcher(None, seg, n).ratio()
            if r > best:
                best = r
            if best >= 1.0:
                break
        if best >= 1.0:
            break
    if best < threshold:
        global_r = difflib.SequenceMatcher(None, h, n).ratio()
        best = max(best, global_r)
    return (best >= threshold, best)

def _choose_surface_variant(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    def conf_or_zero(x):
        c = x.get("confidence")
        try:
            return float(c) if c is not None else 0.0
        except Exception:
            return 0.0
    return max(instances, key=conf_or_zero)

def _collect_by_canonical(p1_tr: List[Dict[str, Any]],
                          p2_tr: List[Dict[str, Any]],
                          p3_tr: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for t in (p1_tr + p2_tr + p3_tr):
        k = t.get("canonical")
        if not k:
            k = canonical_triple_key(t.get("s"), t.get("p"), t.get("o"))
            t["canonical"] = k
        by_key.setdefault(k, []).append(t)
    return by_key

def _rule_a_select(by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
                   input_text: str,
                   threshold: float = 0.90,
                   debug: bool = False) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for _, insts in by_key.items():
        prompts = {t.get("source_prompt") for t in insts}
        if len(prompts) >= 2:
            rep = _choose_surface_variant(insts)
            s_ok, s_score = fuzzy_in(rep["s"], input_text, threshold)
            o_ok, o_score = fuzzy_in(rep["o"], input_text, threshold)
            if s_ok and o_ok:
                selected.append(rep)
                if debug:
                    print(f"  [Rule A PASS] {rep['s']} — {rep['p']} — {rep['o']} "
                          f"(s={s_score:.2f}, o={o_score:.2f}) prompts={sorted(prompts)}")
            else:
                if debug:
                    print(f"  [Rule A FAIL] {rep['s']} — {rep['p']} — {rep['o']} "
                          f"(s={s_score:.2f}, o={o_score:.2f}) prompts={sorted(prompts)}")
    return selected

def _present_in_support_and_input(s: str, support: str, input_text: str, thr: float) -> Tuple[bool, float, float]:
    in_sup, sup_sc = fuzzy_in(s, support, thr)
    in_inp, inp_sc = fuzzy_in(s, input_text, thr)
    return (in_sup and in_inp, sup_sc, inp_sc)

def _rule_b_select(by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
                   input_text: str,
                   threshold: float = 0.90,
                   evidence_cut: float = 0.70,
                   debug: bool = False) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for _, insts in by_key.items():
        prompts = {t.get("source_prompt") for t in insts}
        if len(prompts) != 1:
            continue
        t = insts[0]
        s, p, o = t["s"], t["p"], t["o"]
        sup = t.get("support_text")
        if not sup:
            if debug:
                print(f"  [Rule B REJECT] (no support) {s} — {p} — {o} from {sorted(prompts)}")
            continue
        s_co, _, _ = _present_in_support_and_input(s, sup, input_text, threshold)
        o_co, _, _ = _present_in_support_and_input(o, sup, input_text, threshold)
        p_co, _, _ = _present_in_support_and_input(p, sup, input_text, threshold)
        coloc = 1.0 if (s_co and o_co and p_co) else 0.0
        subj_in_sup, _ = fuzzy_in(s, sup, threshold)
        obj_in_sup, _ = fuzzy_in(o, sup, threshold)
        subj_sup = 1.0 if subj_in_sup else 0.0
        obj_sup = 1.0 if obj_in_sup else 0.0
        sim = max(0.0, min(1.0, jaccard_similarity(sup, input_text)))
        evidence = (0.40 * coloc) + (0.25 * subj_sup) + (0.25 * obj_sup) + (0.10 * sim)
        evidence = max(0.0, min(1.0, evidence))
        if evidence > evidence_cut:
            selected.append(t)
            if debug:
                print(f"  [Rule B PASS] {s} — {p} — {o} | Evidence={evidence:.2f}")
        else:
            if debug:
                print(f"  [Rule B FAIL] {s} — {p} — {o} | Evidence={evidence:.2f}")
    return selected

def _rule_c_select(by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
                   input_text: str,
                   subj_thr: float = 0.80,
                   obj_thr: float = 0.55,
                   evidence_cut: float = 0.40,
                   debug: bool = False) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for _, insts in by_key.items():
        for t in insts:
            if t.get("confidence") is not None or t.get("support_text"):
                continue
            s, p, o = t["s"], t["p"], t["o"]
            s_ok, _ = fuzzy_in(s, input_text, threshold=subj_thr)
            o_ok, _ = fuzzy_in(o, input_text, threshold=obj_thr)
            triple_str = f"{s} {p} {o}"
            sim = max(0.0, min(1.0, jaccard_similarity(triple_str, input_text)))
            s_val = 1.0 if s_ok else 0.0
            o_val = 1.0 if o_ok else 0.0
            evidence = 0.5 * s_val + 0.4 * o_val + 0.1 * sim
            evidence = max(0.0, min(1.0, evidence))
            if evidence >= evidence_cut:
                selected.append(t)
                if debug:
                    print(f"  [Rule C PASS] {s} — {p} — {o} | Evidence_C={evidence:.2f}")
            else:
                if debug:
                    print(f"  [Rule C FAIL] {s} — {p} — {o} | Evidence_C={evidence:.2f}")
    return selected

def evaluate_ids(index_by_id: Dict[str, Dict[str, Any]],
                 out_jsonl_path: str,
                 limit_ids: Optional[int] = None,
                 debug: bool = True) -> None:
    def sort_key(x):
        m = re.findall(r"(\d+)$", x)
        return int(m[0]) if m else x
    all_ids = sorted(index_by_id.keys(), key=sort_key)
    ids = all_ids[:limit_ids] if limit_ids is not None else all_ids
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    with open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for rid in ids:
            rec = index_by_id[rid]
            input_text = rec.get("input_text") or ""
            p1_tr = rec.get("p1", [])
            p2_tr = rec.get("p2", [])
            p3_tr = rec.get("p3", [])
            by_key = _collect_by_canonical(p1_tr, p2_tr, p3_tr)
            if debug:
                print("\n" + "=" * 80)
                print(f"[ID] {rid}")
                print(f"  P1 triples: {len(p1_tr)} | P2: {len(p2_tr)} | P3: {len(p3_tr)}")
                print("  Input text:", input_text)
                print("  → Groups by canonical (s,p,o):")
                for k, insts in by_key.items():
                    prompts = {t.get("source_prompt") for t in insts}
                    print(f"    [GROUP] canonical={k} | prompts={sorted(prompts)} | instances={len(insts)}")
                    for t in insts:
                        print(f"      - from {t.get('source_prompt')}: s='{t['s']}', p='{t['p']}', o='{t['o']}'")
                print("  → Running Rule A …")
            sel_a = _rule_a_select(by_key, input_text, threshold=0.90, debug=debug)
            if debug:
                print("  → Running Rule B …")
            sel_b = _rule_b_select(by_key, input_text, threshold=0.90, evidence_cut=0.70, debug=debug)
            if debug:
                print("  → Running Rule C …")
            sel_c = _rule_c_select(by_key, input_text,
                                   subj_thr=0.80, obj_thr=0.55,
                                   evidence_cut=0.40, debug=debug)
            final_map: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
            for t in (sel_a + sel_b + sel_c):
                k = t.get("canonical")
                final_map.setdefault(k, []).append(t)
            final_triples: List[Dict[str, str]] = []
            seen_spo = set()
            for _, insts in final_map.items():
                rep = _choose_surface_variant(insts)
                spo_key = (rep["s"], rep["p"], rep["o"])
                if spo_key in seen_spo:
                    continue
                seen_spo.add(spo_key)
                final_triples.append({"s": rep["s"], "p": rep["p"], "o": rep["o"]})
            if debug:
                print(f"  → Selected triples: {len(final_triples)}")
                for tt in final_triples:
                    print(f"    [SELECTED] {tt['s']} — {tt['p']} — {tt['o']}")
            out_obj = {
                "id": rid,
                "input_text": input_text,
                "triples": final_triples if final_triples else None
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    if debug:
        print("\n[Evaluator] Done.")
        print(f"Wrote results to: {out_jsonl_path}")

def run_dbpedia_forest_batch(limit_ids: Optional[int] = None,
                             debug: bool = True) -> None:
    print("======== RUN DBPEDIA FOREST BATCH ========")
    os.makedirs(DBPEDIA_FINAL_BASE, exist_ok=True)
    total_files = len(DBPEDIA_FILENAMES)
    for idx, fname in enumerate(DBPEDIA_FILENAMES, start=1):
        domain = extract_domain_from_filename(fname, kb_type="dbpedia")
        if not domain:
            print(f"[ERROR] Could not extract domain from DBpedia filename: {fname} → skipping")
            continue
        p1_path = os.path.join(DBPEDIA_P1_BASE, fname)
        p2_path = os.path.join(DBPEDIA_P2_BASE, fname)
        p3_path = os.path.join(DBPEDIA_P3_BASE, fname)
        out_name = fname.replace("_output.jsonl", "_filtered_output.jsonl") if fname.endswith("_output.jsonl") else fname
        out_path = os.path.join(DBPEDIA_FINAL_BASE, out_name)
        print("\n" + "#" * 70)
        print(f"[DBPEDIA {idx}/{total_files}] Ontology file: {fname}")
        print(f"  Domain    : {domain}")
        print(f"  P1 file   : {p1_path}")
        print(f"  P2 file   : {p2_path}")
        print(f"  P3 file   : {p3_path}")
        print(f"  Final out : {out_path}")
        P1 = load_prompt_outputs_strict(p1_path, "P1", domain, kb_type="dbpedia")
        P2 = load_prompt_outputs_strict(p2_path, "P2", domain, kb_type="dbpedia")
        P3 = load_prompt_outputs_strict(p3_path, "P3", domain, kb_type="dbpedia")
        INDEX_BY_ID = build_id_index(P1, P2, P3)
        summarize_loaded(INDEX_BY_ID)
        evaluate_ids(INDEX_BY_ID, out_jsonl_path=out_path, limit_ids=limit_ids, debug=debug)
    print("\n======== DBPEDIA FOREST BATCH DONE ========")

def run_wikidata_forest_batch(limit_ids: Optional[int] = None,
                              debug: bool = True) -> None:
    print("======== RUN WIKIDATA FOREST BATCH ========")
    os.makedirs(WIKI_FINAL_BASE, exist_ok=True)
    total_files = len(WIKI_FILENAMES)
    for idx, fname in enumerate(WIKI_FILENAMES, start=1):
        domain = extract_domain_from_filename(fname, kb_type="wikidata")
        if not domain:
            print(f"[ERROR] Could not extract domain from Wikidata filename: {fname} → skipping")
            continue
        p1_path = os.path.join(WIKI_P1_BASE, fname)
        p2_path = os.path.join(WIKI_P2_BASE, fname)
        p3_path = os.path.join(WIKI_P3_BASE, fname)
        out_name = fname.replace("_output.jsonl", "_filtered_output.jsonl") if fname.endswith("_output.jsonl") else fname
        out_path = os.path.join(WIKI_FINAL_BASE, out_name)
        print("\n" + "#" * 70)
        print(f"[WIKIDATA {idx}/{total_files}] Ontology file: {fname}")
        print(f"  Domain    : {domain}")
        print(f"  P1 file   : {p1_path}")
        print(f"  P2 file   : {p2_path}")
        print(f"  P3 file   : {p3_path}")
        print(f"  Final out : {out_path}")
        P1 = load_prompt_outputs_strict(p1_path, "P1", domain, kb_type="wikidata")
        P2 = load_prompt_outputs_strict(p2_path, "P2", domain, kb_type="wikidata")
        P3 = load_prompt_outputs_strict(p3_path, "P3", domain, kb_type="wikidata")
        INDEX_BY_ID = build_id_index(P1, P2, P3)
        summarize_loaded(INDEX_BY_ID)
        evaluate_ids(INDEX_BY_ID, out_jsonl_path=out_path, limit_ids=limit_ids, debug=debug)
    print("\n======== WIKIDATA FOREST BATCH DONE ========")

if __name__ == "__main__":
    #run_dbpedia_forest_batch(limit_ids=None, debug=False)
    run_wikidata_forest_batch(limit_ids=None, debug=False)