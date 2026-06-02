#
import os
import re
import csv
import json
import time
import html
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
LIST_URL = ("https://company.sbb.ch/content/internet/corporate/de/"
            "jobs-karriere/jobs/job-suche/jcr:content/parmain/"
            "jobfilter.results.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (HSLU-StudentProject; educational use; skill-extraction)",
}

TARGET_ADS = 30            # how many cleaned German ads to keep
REQUEST_DELAY = 1.5        # seconds between detail requests (politeness)
MIN_CHARS = 200            # discard ads with too little text
ONLY_GERMAN = True         # keep only ads with addressCountry == "Schweiz"
OUTPUT_DIR = "data"
TXT_DIR = os.path.join(OUTPUT_DIR, "raw_txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("crawler")


# --------------------------------------------------------------------
# Spec step: Ethical Considerations -> check robots.txt
# --------------------------------------------------------------------
def robots_allows(url: str, user_agent: str = "*") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as e:
        log.warning("robots.txt not readable (%s) -> proceeding cautiously", e)
        return True
    allowed = rp.can_fetch(user_agent, url)
    log.info("robots.txt: access to %s is %s", url, "ALLOWED" if allowed else "DISALLOWED")
    return allowed


# --------------------------------------------------------------------
# Spec step: Data Cleaning
# --------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?i)<\s*br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|li|ul|ol|h[1-6])\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[\s]*", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


# --------------------------------------------------------------------
# Spec step: Navigation & Extraction
# --------------------------------------------------------------------
def fetch_job_list() -> list:
    r = requests.get(LIST_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    entries = r.json()
    jobs = []
    for e in entries:
        link = (e.get("links", {}) or {}).get("directlink", "")
        if not link:
            continue
        attrs = e.get("attributes", {}) or {}
        country = (attrs.get("65", [""]) or [""])[0]   # "Schweiz" / "Suisse" / "Svizzera"
        jobs.append({
            "title": e.get("title", ""),
            "link": link,
            "country": country,
        })
    log.info("Job feed: %d entries total.", len(jobs))
    return jobs


def parse_detail(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        log.warning("  detail -> HTTP %d (skipped): %s", r.status_code, url)
        return {}
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the JSON-LD block whose @type is JobPosting
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("@type") == "JobPosting":
            return data
    return {}


# --------------------------------------------------------------------
# Spec step: Data Storage
# --------------------------------------------------------------------
def save_outputs(jobs: list):
    os.makedirs(TXT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "crawled_jobs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUTPUT_DIR, "crawled_jobs.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["job_id", "titel", "arbeitgeber", "ort",
                           "land", "link", "content_clean"])
        writer.writeheader()
        for j in jobs:
            writer.writerow(j)

    for j in jobs:
        with open(os.path.join(TXT_DIR, f"{j['job_id']}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(j["content_clean"])

    log.info("Saved: %s | %s | %d text files in %s/",
             json_path, csv_path, len(jobs), TXT_DIR)


# --------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not robots_allows(LIST_URL, user_agent=HEADERS["User-Agent"]):
        log.error("robots.txt disallows access. Aborting.")
        return

    # 1) Get the job list
    try:
        listing = fetch_job_list()
    except Exception as e:
        log.error("Failed to fetch job list: %s", e)
        return

    # Optionally keep only German-language ads (country == "Schweiz")
    if ONLY_GERMAN:
        listing = [j for j in listing if j["country"] == "Schweiz"]
        log.info("After German-only filter: %d ads.", len(listing))

    # 2) Fetch details, extract JSON-LD, clean, collect
    jobs = []
    for i, item in enumerate(listing, start=1):
        if len(jobs) >= TARGET_ADS:
            break

        data = parse_detail(item["link"])
        if not data:
            log.info("  [%d] no JSON-LD -> skipped: %s", i, item["title"][:50])
            time.sleep(REQUEST_DELAY)
            continue

        # Build the full ad text from the structured fields.
        # 'qualifications' is the skills/requirements part -> most relevant
        parts = [
            clean_text(data.get("responsibilities", "")),
            clean_text(data.get("qualifications", "")),
        ]
        # fall back to 'description' if the split fields are empty
        combined = "\n\n".join(p for p in parts if p).strip()
        if not combined:
            combined = clean_text(data.get("description", ""))

        if len(combined) < MIN_CHARS:
            log.info("  [%d] too short -> skipped: %s", i, item["title"][:50])
            time.sleep(REQUEST_DELAY)
            continue

        addr = (data.get("jobLocation", {}) or {}).get("address", {}) or {}
        idx = len(jobs) + 1
        jobs.append({
            "job_id": f"crawl_{idx:03d}",
            "titel": data.get("title", item["title"]),
            "arbeitgeber": (data.get("hiringOrganization", {}) or {}).get("name", "SBB"),
            "ort": addr.get("addressLocality", ""),
            "land": addr.get("addressCountry", item["country"]),
            "link": item["link"],
            "content_clean": combined,   # same field name as annotated.json!
        })
        log.info("  [%d/%d] OK: %s (%d chars)",
                 len(jobs), TARGET_ADS, data.get("title", "")[:50], len(combined))
        time.sleep(REQUEST_DELAY)        # politeness between detail requests

    # 3) Save
    if jobs:
        save_outputs(jobs)
        log.info("DONE: collected and saved %d job ads.", len(jobs))
    else:
        log.error("No usable ads collected.")


if __name__ == "__main__":
    main()
