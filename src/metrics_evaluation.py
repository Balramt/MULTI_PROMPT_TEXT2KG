from __future__ import annotations

import os
import json
import re
from typing import List, Dict, Tuple, Optional

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

DBPEDIA_BASE_SYS = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/evaluator_filtered_output/dbpedia1/"
DBPEDIA_BASE_GT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ground_truth/dbpedia/"
DBPEDIA_BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/dbpedia/"
DBPEDIA_BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/metrics_evaluation/dbpedia1/"
DBPEDIA_AVG_FILE = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/metrics_evaluation/Overall_average_dbpedia1.jsonl"

DBPEDIA_FILENAMES = [
    "ont_1_university_filtered_output.jsonl",
    "ont_2_musicalwork_filtered_output.jsonl",
    "ont_3_airport_filtered_output.jsonl",
    "ont_4_building_filtered_output.jsonl",
    "ont_5_athlete_filtered_output.jsonl",
    "ont_6_politician_filtered_output.jsonl",
    "ont_7_company_filtered_output.jsonl",
    "ont_8_celestialbody_filtered_output.jsonl",
    "ont_9_astronaut_filtered_output.jsonl",
    "ont_10_comicscharacter_filtered_output.jsonl",
    "ont_11_meanoftransportation_filtered_output.jsonl",
    "ont_12_monument_filtered_output.jsonl",
    "ont_13_food_filtered_output.jsonl",
    "ont_14_writtenwork_filtered_output.jsonl",
    "ont_15_sportsteam_filtered_output.jsonl",
    "ont_16_city_filtered_output.jsonl",
    "ont_17_artist_filtered_output.jsonl",
    "ont_18_scientist_filtered_output.jsonl",
    "ont_19_film_filtered_output.jsonl",
]

WIKI_BASE_SYS = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/evaluator_filtered_output/wikidata/"
WIKI_BASE_GT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ground_truth/wikidata/"
WIKI_BASE_ONTO = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/input/ontology/old_ontology/wikidata/"
WIKI_BASE_OUT = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/metrics_evaluation/wikidata1/"
WIKI_AVG_FILE = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp2_thesis/data/output/metrics_evaluation/overall_average_wikidata1.jsonl"

WIKI_FILENAMES = [
    "ont_1_movie_filtered_output.jsonl",
    "ont_2_music_filtered_output.jsonl",
    "ont_3_sport_filtered_output.jsonl",
    "ont_4_book_filtered_output.jsonl",
    "ont_5_military_filtered_output.jsonl",
    "ont_6_computer_filtered_output.jsonl",
    "ont_7_space_filtered_output.jsonl",
    "ont_8_politics_filtered_output.jsonl",
    "ont_9_nature_filtered_output.jsonl",
    "ont_10_culture_filtered_output.jsonl",
]

def read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def save_jsonl(items: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def append_jsonl(item: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def lemmatize_and_normalize(text: str, lemmatizer: WordNetLemmatizer) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = text.strip()
    if not text:
        return ""
    tokens = word_tokenize(text)
    lemmatized = "".join(lemmatizer.lemmatize(tok) for tok in tokens)
    return lemmatized

def normalize_triple(
    sub_label: str,
    rel_label: str,
    obj_label: str,
    lemmatizer: WordNetLemmatizer,
    stem_rel: bool = False,
) -> str:
    sub_n = lemmatize_and_normalize(sub_label, lemmatizer)
    obj_n = lemmatize_and_normalize(obj_label, lemmatizer)

    rel_label_clean = rel_label.lower().replace("_", " ")
    rel_label_clean = re.sub(r"[^a-z0-9 ]", "", rel_label_clean).strip()

    if stem_rel:
        rel_label_clean = " ".join(lemmatizer.lemmatize(w) for w in word_tokenize(rel_label_clean))

    rel_label_clean = re.sub(r"\s+", "", rel_label_clean)
    return f"{sub_n}{rel_label_clean}{obj_n}"

def calculate_precision_recall_f1(gold: set, pred: set) -> Tuple[float, float, float]:
    if not pred:
        return 0.0, 0.0, 0.0
    correct = len(gold & pred)
    p = correct / len(pred)
    r = correct / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

def extract_system_triples_from_record(rec: dict) -> List[List[str]]:
    if not isinstance(rec, dict):
        return []
    triples_block = rec.get("triples")
    if not isinstance(triples_block, list):
        return []
    cleaned: List[List[str]] = []
    for item in triples_block:
        if (
            isinstance(item, dict) and
            all(k in item for k in ("s", "p", "o")) and
            all(isinstance(item[k], str) for k in ("s", "p", "o"))
        ):
            cleaned.append([item["s"], item["p"], item["o"]])
    return cleaned

def get_ontology_conformance(
    ontology: Dict,
    triples: List[List[str]],
    lemmatizer: WordNetLemmatizer,
) -> Tuple[float, float]:
    if not triples:
        return 1.0, 0.0

    ont_rels = {lemmatize_and_normalize(rel.get("label", ""), lemmatizer) for rel in ontology.get("relations", [])}
    num_conformant = 0
    for _sub, rel, _obj in triples:
        if lemmatize_and_normalize(rel, lemmatizer) in ont_rels:
            num_conformant += 1

    oc = num_conformant / len(triples)
    return oc, 1.0 - oc

def get_subject_object_hallucinations(
    ontology: Dict,
    sentence: str,
    triples: List[List[str]],
    lemmatizer: WordNetLemmatizer,
) -> Tuple[float, float]:
    if not triples:
        return 0.0, 0.0

    concepts = ontology.get("concepts", [])
    concept_labels = " ".join(c.get("label", "") for c in concepts if isinstance(c, dict))
    extended_sentence = (sentence or "") + " " + concept_labels
    normalized_context = lemmatize_and_normalize(extended_sentence, lemmatizer)

    subj_h, obj_h = 0, 0
    for sub, _rel, obj in triples:
        norm_sub = lemmatize_and_normalize(sub, lemmatizer)
        norm_obj = lemmatize_and_normalize(obj, lemmatizer)

        if norm_sub and norm_sub not in normalized_context:
            subj_h += 1
        if norm_obj and norm_obj not in normalized_context:
            obj_h += 1

    total = len(triples)
    return subj_h / total, obj_h / total

DBPEDIA_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_(?:filtered_)?output\.jsonl$")
WIKIDATA_PATTERN = re.compile(r"^ont_(\d+)_([a-zA-Z0-9]+)_(?:filtered_)?output\.jsonl$")

def make_dbpedia_paths(filename: str) -> Tuple[str, str, str, str, str]:
    m = DBPEDIA_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Unexpected DBpedia filename: {filename}")
    idx, cat = m.groups()
    tag = f"ont_{idx}_{cat}"
    sys_path = os.path.join(DBPEDIA_BASE_SYS, filename)
    gt_path = os.path.join(DBPEDIA_BASE_GT, f"{tag}_ground_truth.jsonl")
    onto_path = os.path.join(DBPEDIA_BASE_ONTO, f"{idx}_{cat}_ontology.json")
    out_path = os.path.join(DBPEDIA_BASE_OUT, f"{tag}_eval_sentences.jsonl")
    return sys_path, gt_path, onto_path, out_path, tag

def make_wikidata_paths(filename: str) -> Tuple[str, str, str, str, str]:
    m = WIKIDATA_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Unexpected Wikidata filename: {filename}")
    idx, cat = m.groups()
    tag = f"ont_{idx}_{cat}"
    sys_path = os.path.join(WIKI_BASE_SYS, filename)
    gt_path = os.path.join(WIKI_BASE_GT, f"{tag}_ground_truth.jsonl")
    onto_path = os.path.join(WIKI_BASE_ONTO, f"{idx}_{cat}_ontology.json")
    out_path = os.path.join(WIKI_BASE_OUT, f"{tag}_eval_sentences.jsonl")
    return sys_path, gt_path, onto_path, out_path, tag

def evaluate_one_file(
    sys_path: str,
    gt_path: str,
    onto_path: str,
    out_path: str,
    avg_out_file: str,
    tag: str,
    id_mode: str = "all",
    id_start: Optional[int] = None,
    id_end: Optional[int] = None,
    debug: bool = False,
) -> None:
    lemmatizer = WordNetLemmatizer()

    system_list = read_jsonl(sys_path)
    gt_list = read_jsonl(gt_path)
    ontology = read_json(onto_path)

    system_by_id = {rec["id"]: rec for rec in system_list if isinstance(rec, dict) and "id" in rec}
    gt_by_id = {rec["id"]: rec for rec in gt_list if isinstance(rec, dict) and "id" in rec}

    def id_num(id_str: str) -> int:
        m = re.search(r"(\d+)$", id_str)
        return int(m.group(1)) if m else -1

    selected_ids = sorted(gt_by_id.keys(), key=lambda x: id_num(x))
    if id_mode == "range" and id_start is not None and id_end is not None:
        selected_ids = [i for i in selected_ids if id_start <= id_num(i) <= id_end]

    t_p = t_r = t_f1 = 0.0
    t_onto = t_rel_h = 0.0
    t_sub_h = t_obj_h = 0.0
    per_sent: List[Dict] = []

    counted = 0

    for sent_id in selected_ids:
        gt_rec = gt_by_id[sent_id]
        gt_triples_block = gt_rec.get("triples", [])
        if not gt_triples_block:
            continue

        if sent_id not in system_by_id:
            continue

        sentence = gt_rec.get("sent", "")

        gt_triples = [[tr["sub"], tr["rel"], tr["obj"]] for tr in gt_triples_block]

        sys_rec = system_by_id[sent_id]
        sys_triples = extract_system_triples_from_record(sys_rec)

        gt_relations = {lemmatize_and_normalize(tr[1], lemmatizer) for tr in gt_triples}
        sys_triples_f = [
            [sub, rel, obj]
            for (sub, rel, obj) in sys_triples
            if lemmatize_and_normalize(rel, lemmatizer) in gt_relations
        ]

        norm_gt = {normalize_triple(sub, rel, obj, lemmatizer) for (sub, rel, obj) in gt_triples}
        norm_sys = {normalize_triple(sub, rel, obj, lemmatizer) for (sub, rel, obj) in sys_triples_f}

        p, r, f1 = calculate_precision_recall_f1(norm_gt, norm_sys)

        onto_conf, rel_halluc = get_ontology_conformance(ontology, sys_triples, lemmatizer)
        sub_h, obj_h = get_subject_object_hallucinations(ontology, sentence, sys_triples, lemmatizer)

        per_sent.append({
            "id": sent_id,
            "precision": f"{p:.2f}",
            "recall": f"{r:.2f}",
            "f1": f"{f1:.2f}",
            "onto_conf": f"{onto_conf:.2f}",
            "rel_halluc": f"{rel_halluc:.2f}",
            "sub_halluc": f"{sub_h:.2f}",
            "obj_halluc": f"{obj_h:.2f}",
            "llm_triples": sys_triples,
            "filtered_llm_triples": sys_triples_f,
            "gt_triples": gt_triples,
            "sent": sentence,
        })

        t_p += p
        t_r += r
        t_f1 += f1
        t_onto += onto_conf
        t_rel_h += rel_halluc
        t_sub_h += sub_h
        t_obj_h += obj_h
        counted += 1

    save_jsonl(per_sent, out_path)

    denom = counted if counted > 0 else 1
    avg_metrics = {
        "onto": tag,
        "type": "all_test_cases",
        "avg_precision": f"{t_p / denom:.2f}",
        "avg_recall": f"{t_r / denom:.2f}",
        "avg_f1": f"{t_f1 / denom:.2f}",
        "avg_onto_conf": f"{t_onto / denom:.2f}",
        "avg_sub_halluc": f"{t_sub_h / denom:.2f}",
        "avg_rel_halluc": f"{t_rel_h / denom:.2f}",
        "avg_obj_halluc": f"{t_obj_h / denom:.2f}",
    }
    append_jsonl(avg_metrics, avg_out_file)

def run_dbpedia_eval_batch(
    n_limit: Optional[int] = None,
    id_mode: str = "all",
    id_start: Optional[int] = None,
    id_end: Optional[int] = None,
    debug: bool = False,
) -> None:
    if os.path.exists(DBPEDIA_AVG_FILE):
        open(DBPEDIA_AVG_FILE, "w", encoding="utf-8").close()

    files_to_run = DBPEDIA_FILENAMES if n_limit is None else DBPEDIA_FILENAMES[:n_limit]

    for filename in files_to_run:
        sys_path, gt_path, onto_path, out_path, tag = make_dbpedia_paths(filename)
        evaluate_one_file(
            sys_path=sys_path,
            gt_path=gt_path,
            onto_path=onto_path,
            out_path=out_path,
            avg_out_file=DBPEDIA_AVG_FILE,
            tag=tag,
            id_mode=id_mode,
            id_start=id_start,
            id_end=id_end,
            debug=debug,
        )

def run_wikidata_eval_batch(
    n_limit: Optional[int] = None,
    id_mode: str = "all",
    id_start: Optional[int] = None,
    id_end: Optional[int] = None,
    debug: bool = False,
) -> None:
    if os.path.exists(WIKI_AVG_FILE):
        open(WIKI_AVG_FILE, "w", encoding="utf-8").close()

    files_to_run = WIKI_FILENAMES if n_limit is None else WIKI_FILENAMES[:n_limit]

    for filename in files_to_run:
        sys_path, gt_path, onto_path, out_path, tag = make_wikidata_paths(filename)
        evaluate_one_file(
            sys_path=sys_path,
            gt_path=gt_path,
            onto_path=onto_path,
            out_path=out_path,
            avg_out_file=WIKI_AVG_FILE,
            tag=tag,
            id_mode=id_mode,
            id_start=id_start,
            id_end=id_end,
            debug=debug,
        )

if __name__ == "__main__":
    #run_dbpedia_eval_batch()
    run_wikidata_eval_batch()