
import os
import re
import csv
import json
import time

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ====================================================================
# Configuration
# ====================================================================
MODEL_NAME = "bert-base-multilingual-cased"
MODEL_PATH = os.path.join("model", "best_student_model.pt")
TARGET_LABEL = "Fähigkeiten und Inhalte"
MAX_LEN = 510

SOURCE = "crawled"                       # "crawled" or "testset"
CRAWLED_PATH = os.path.join("data", "crawled_jobs.json")
TESTSET_PATH = os.path.join("data", "test_dataset.pt")
MAX_ADS = 20                             # how many ads to process
OUTPUT_DIR = "data"

GEMINI_MODEL = "gemini-2.5-flash-lite"        # free-tier model
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
API_DELAY = 4.5                          # seconds between API calls (rate limit)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# Part A: BERT zone inference
# ====================================================================
_model = None
_tokenizer = None
_id2label = None


def load_bert(model_path: str = MODEL_PATH):
    global _model, _tokenizer, _id2label
    if _model is not None:
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    _id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
    num_labels = checkpoint.get("num_labels", len(_id2label))

    _model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=_id2label,
        label2id={v: k for k, v in _id2label.items()},
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(device)
    _model.eval()

    _tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f"BERT loaded: {num_labels} labels on {device}")


def _detok(tokens):
    out = ""
    for t in tokens:
        if t.startswith("##"):
            out += t[2:]
        else:
            out += (" " if out else "") + t
    return out.strip()


def _spans_to_text(pairs):
    segments, current = [], []
    for tok, lab in pairs:
        if lab == TARGET_LABEL:
            current.append(tok)
        elif current:
            segments.append(current); current = []
    if current:
        segments.append(current)
    return [t for t in (_detok(s) for s in segments) if t]


def predict_from_text(text: str):
    load_bert()
    enc = _tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
    preds = logits.argmax(dim=-1)[0].tolist()
    tokens = _tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    pairs = [(t, _id2label[p]) for t, p in zip(tokens, preds)
             if t not in ("[CLS]", "[SEP]", "[PAD]")]
    return _spans_to_text(pairs)


def predict_from_ids(input_ids, attention_mask):
    load_bert()
    ids = torch.as_tensor(input_ids).unsqueeze(0).to(device)
    attn = torch.as_tensor(attention_mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(input_ids=ids, attention_mask=attn).logits
    preds = logits.argmax(dim=-1)[0].tolist()
    tokens = _tokenizer.convert_ids_to_tokens(ids[0].tolist())
    pairs = [(t, _id2label[p]) for t, p, a in zip(tokens, preds, attn[0].tolist())
             if a == 1 and t not in ("[CLS]", "[SEP]", "[PAD]")]
    return _spans_to_text(pairs)


# ====================================================================
# Part B: Gemini skill extraction
# ====================================================================
_gemini = None

PROMPT_TEMPLATE = """You are an expert HR analyst specializing in skill identification.
The text below is an excerpt from the 'Skills and Content' section of a job advertisement (in German).

Task: Identify and extract all relevant professional skills mentioned in the text.

Rules:
- Each skill must be a meaningful phrase of at least two (2) and at most five (5) words.
- Do not extract single words.
- Focus on technical and professional skills.
- Provide the output ONLY as a JSON list of strings, with no extra text.

Examples:
Input: "Requires strong Java programming and experience with Spring Boot."
Output: ["Java programming", "Spring Boot experience"]

Input: "Du besitzt fundierte Kenntnisse in SAP, insbesondere in den Modulen CO und FI."
Output: ["SAP Kenntnisse", "Module CO und FI"]

Now extract the skills from this text:
\"\"\"{zone_text}\"\"\"
"""


def init_gemini():
    global _gemini
    if _gemini is not None:
        return True
    if not GENAI_AVAILABLE:
        print("google-generativeai not installed -> skipping Gemini step.")
        return False
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set -> skipping Gemini step.")
        return False
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini = genai.GenerativeModel(GEMINI_MODEL)
    return True


def parse_skills(raw: str):
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"^```(?:json|python)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    skills = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            skills = [str(s) for s in parsed]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip().lstrip("-*•").strip().strip('"\'').strip(",")
            if line:
                skills.append(line)

    cleaned, seen = [], set()
    for s in skills:
        s = s.strip().strip('"\'').strip()
        wc = len(s.split())
        if 2 <= wc <= 5 and s.lower() not in seen:
            seen.add(s.lower())
            cleaned.append(s)
    return cleaned


def extract_skills(zone_text: str, max_retries: int = 3, base_delay: float = 2.0):
    if not init_gemini():
        return []
    prompt = PROMPT_TEMPLATE.format(zone_text=zone_text)

    for attempt in range(1, max_retries + 1):
        try:
            resp = _gemini.generate_content(prompt)
            return parse_skills(resp.text)
        except Exception as e:                   # network / quota / invalid response
            if attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))   # 2s, 4s, 8s ...
                print(f"  Gemini call failed ({type(e).__name__}); "
                      f"retry {attempt}/{max_retries - 1} in {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  Gemini call failed after {max_retries} attempts: {e}")
    return []


# ====================================================================
# Part C: Data loading
# ====================================================================
def load_crawled():
    with open(CRAWLED_PATH, encoding="utf-8") as f:
        ads = json.load(f)
    for ad in ads[:MAX_ADS]:
        yield ad["job_id"], ad["content_clean"]


def load_testset():
    ds = torch.load(TESTSET_PATH, weights_only=False)
    for i in range(min(MAX_ADS, len(ds))):
        input_ids, _labels, attn = ds[i]
        yield f"test_{i:03d}", input_ids, attn


# ====================================================================
# Part D: Pipeline
# ====================================================================
def run():
    results = []

    if SOURCE == "crawled":
        iterator = load_crawled()
    elif SOURCE == "testset":
        iterator = load_testset()
    else:
        raise ValueError(f"Unknown SOURCE: {SOURCE}")

    for item in iterator:
        if SOURCE == "crawled":
            job_id, text = item
            segments = predict_from_text(text)
        else:
            job_id, ids, attn = item
            segments = predict_from_ids(ids, attn)

        # Spec: handle ads where BERT detects no skills zone
        if not segments:
            print(f"{job_id}: no '{TARGET_LABEL}' zone detected -> skipped")
            results.append({"job_id": job_id, "zone_text": "", "skills": []})
            continue

        zone_text = " ".join(segments)
        skills = extract_skills(zone_text)
        print(f"{job_id}: {len(skills)} skills extracted")
        results.append({
            "job_id": job_id,
            "zone_text": zone_text,
            "skills": skills,
        })
        time.sleep(API_DELAY)                    # respect rate limit

    save_results(results)


def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "extracted_skills.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "extracted_skills.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job_id", "skill"])
        for r in results:
            for skill in r["skills"]:
                writer.writerow([r["job_id"], skill])

    total = sum(len(r["skills"]) for r in results)
    print(f"\nSaved {json_path} and {csv_path}")
    print(f"{len(results)} ads processed, {total} skills extracted in total.")


if __name__ == "__main__":
    run()
