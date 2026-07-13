

import os
import json
import hashlib
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "data/image_summary_cache.json"

llm_vision = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # cheap + fast is enough for this step


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _hash_file(path: str) -> str:
    """Used as the cache key so identical images (e.g. repeated logos) are only summarized once."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def summarize_image(image_path: str) -> str:
    """Returns a plain-text description of one image. Uses cache when possible."""
    cache = _load_cache()
    file_hash = _hash_file(image_path)

    if file_hash in cache:
        return cache[file_hash]

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Describe this image in 2-3 sentences. "
        "If it's a chart, table, or nutrition label, mention the specific "
        "numbers/labels visible. Be factual, no guessing."
    )

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
    ])

    response = llm_vision.invoke([message])
    summary = response.content.strip()

    cache[file_hash] = summary
    _save_cache(cache)

    return summary


def summarize_images(image_entries: list) -> list:
    """
    image_entries: [{"path": "...", "page": 3}, ...]
    Returns the same list with an added "summary" field for each image.
    """
    results = []
    for entry in image_entries:
        summary = summarize_image(entry["path"])
        results.append({**entry, "summary": summary})
    return results