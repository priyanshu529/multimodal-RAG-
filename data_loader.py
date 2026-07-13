from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import os
import shutil
import base64
import hashlib
import json
from langchain_core.documents import Document
from image_summarizer import summarize_image


def load_pdf_elements(pdf_path: str, image_output_dir: str = "extracted_images") -> list[Document]:

    
    os.makedirs(image_output_dir, exist_ok=True)

    raw_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=image_output_dir,
    )

    documents = []
    text_like_elements = []  # collected separately, chunked by heading at the end

    for el in raw_elements:
        el_type = type(el).__name__
        page = el.metadata.page_number

        if el_type == "Table":
            table_html = el.metadata.text_as_html or str(el)
            preview = table_html.replace("<table>", "").replace("</table>", "")[:300]
            documents.append(Document(
                page_content=f"Table data: {preview}",
                metadata={"type": "table", "page": page, "raw_content": table_html},
            ))

        elif el_type == "Image" and el.metadata.image_path:
            summary = summarize_image(el.metadata.image_path)
            documents.append(Document(
                page_content=summary,
                metadata={"type": "image", "page": page, "raw_content": el.metadata.image_path},
            ))

        else:
            text_like_elements.append(el)

    # Group remaining text under its nearest heading instead of splitting by character count
    for chunk in chunk_by_title(text_like_elements, max_characters=2000,overlap=400):
        content = str(chunk).strip()
        if content:
            documents.append(Document(
                page_content=content,
                metadata={"type": "text", "page": chunk.metadata.page_number, "raw_content": content},
            ))

    return documents