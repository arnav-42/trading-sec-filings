"""
Clean raw HTML → plain text and store JSONL ready for LLM.
"""
import json
import re
from pathlib import Path
from typing import Dict

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import CFG, LOG, ensure_dir

CLEAN_RE = re.compile(r"\s+")

def extract_relevant_sections(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    # Naive slice: keep everything; in production you'd isolate MD&A / Risk Factors
    return CLEAN_RE.sub(" ", text).strip()

def main():
    raw_dir = Path(CFG["ingest"]["save_dir"])
    index_file = raw_dir / "index.json"
    with open(index_file, encoding="utf-8") as fh:
        index = json.load(fh)

    out_path = Path(CFG["preprocess"]["output_file"])
    ensure_dir(str(out_path))

    with open(out_path, "w", encoding="utf-8") as out_fh:
        for meta in tqdm(index, desc="Preprocessing"):
            html = Path(meta["file"]).read_text(encoding="utf-8", errors="ignore")
            cleaned = extract_relevant_sections(html)
            rec: Dict = {**meta, "clean_text": cleaned[:200_000]}  # truncate huge filings
            out_fh.write(json.dumps(rec) + "\n")

    LOG.info("Preprocessed → %s", out_path)

if __name__ == "__main__":
    main()
