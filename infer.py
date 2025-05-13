"""
Batch call LLM for sentiment classification.
Currently uses OpenAI-compatible SDK; switch provider via config.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

import aiofiles
import openai
from tqdm.asyncio import tqdm

from utils import CFG, LOG, chunked, ensure_dir

PROVIDER = CFG["llm"]["provider"]
MODEL = CFG["llm"]["model"]

if PROVIDER == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    openai.api_key = os.getenv("GROQ_API_KEY")
    openai.base_url = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "You are a senior equity analyst. "
    "Classify the overall sentiment of the SEC filing text that follows "
    "as a float in range [-1, 1] where -1 is very negative, +1 very positive. "
    "Respond ONLY with JSON: {\"sentiment\": <float>, \"confidence\": <float>}."
)

async def call_llm(texts: List[str]) -> List[Dict]:
    return [
        await openai.ChatCompletion.acreate(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk[:16_000]},  # token safety
            ],
        )
        for chunk in texts
    ]

async def process_batch(batch: List[Dict]) -> List[Dict]:
    texts = [b["clean_text"] for b in batch]
    resps = await call_llm(texts)
    outputs: List[Dict] = []
    for meta, resp in zip(batch, resps):
        content = resp.choices[0].message.content
        try:
            j = json.loads(content)
            outputs.append({**meta, **j})
        except json.JSONDecodeError:
            LOG.error("Bad JSON from LLM: %s...", content[:60])
    return outputs

async def main():
    src_path = Path(CFG["preprocess"]["output_file"])
    dst_path = Path(CFG["llm"]["output_file"])
    ensure_dir(str(dst_path))

    with open(src_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]

    results: List[Dict] = []
    tasks = [process_batch(batch) for batch in chunked(records, CFG["llm"]["batch_size"])]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM"):
        results.extend(await coro)

    async with aiofiles.open(dst_path, "w", encoding="utf-8") as fh:
        for rec in results:
            await fh.write(json.dumps(rec) + "\n")
    LOG.info("Inference complete → %s", dst_path)

if __name__ == "__main__":
    asyncio.run(main())
