"""
build_documents.py

Job: Turn text, table, and image elements into a single list of LangChain
Document objects, ready to be embedded and indexed.

Key idea (same pattern as your calorie tracker):
  - page_content = SHORT text used for SEARCHING (semantic matching)
  - metadata     = FULL raw content used for ANSWERING (sent to the LLM later)

This way, search matches on simple meaning, but the LLM always gets the
real table/image data, not a lossy summary.
"""

from langchain_core.documents import Document


def build_documents(text_elements: list, table_elements: list, summarized_images: list) -> list:
    documents = []

    # --- text chunks: page_content IS the content, nothing fancy needed ---
    for el in text_elements:
        documents.append(Document(
            page_content=el["content"],
            metadata={"type": "text", "page": el["page"], "raw_content": el["content"]},
        ))

    # --- table chunks: search on a short version, answer with full HTML ---
    for el in table_elements:
        # Strip HTML tags just to make a lighter-weight text for embedding/search
        plain_preview = el["content"].replace("<table>", "").replace("</table>", "")[:300]
        documents.append(Document(
            page_content=f"Table data: {plain_preview}",
            metadata={"type": "table", "page": el["page"], "raw_content": el["content"]},
        ))

    # --- image chunks: search on the vision-model summary, answer with image path ---
    for el in summarized_images:
        documents.append(Document(
            page_content=el["summary"],
            metadata={"type": "image", "page": el["page"], "raw_content": el["path"]},
        ))

    return documents
