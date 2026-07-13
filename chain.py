from prompt import ANSWER_PROMPT
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def format_context(retrieved_docs: list) -> str:
    """
    Builds the final text block sent to the LLM.
    IMPORTANT: we use metadata['raw_content'], not page_content (the summary).
    The LLM should answer from the real data, not a lossy search-time summary.
    """
    parts = []
    for doc in retrieved_docs:
        doc_type = doc.metadata.get("type", "text")
        page = doc.metadata.get("page", "?")
        raw = doc.metadata.get("raw_content", doc.page_content)

        if doc_type == "table":
            parts.append(f"[Table from page {page}]\n{raw}")
        elif doc_type == "image":
            # raw here is a file path; we pass the earlier vision summary as the
            # text stand-in since the LLM in this chain is text-only.
            parts.append(f"[Image from page {page}, description]\n{doc.page_content}")
        else:
            parts.append(f"[Text from page {page}]\n{raw}")

    return "\n\n".join(parts)


def answer_question(question: str, retriever) -> str:
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content
