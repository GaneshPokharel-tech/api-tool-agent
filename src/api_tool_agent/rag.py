import math
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GENERATION_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

CANDIDATE_K = 8
FINAL_K = 4

SEMANTIC_WEIGHT = 0.85
LEXICAL_WEIGHT = 0.15
MMR_LAMBDA = 0.75
MIN_RETRIEVAL_SCORE = 0.20



# ---------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")

gemini_client = genai.Client(api_key=api_key)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ---------------------------------------------------------
# In-Memory RAG Store
# ---------------------------------------------------------

RAG_STORE: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalize whitespace while keeping the text readable.
    """

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_document_pages(
    filename: str,
    file_bytes: bytes,
) -> list[dict[str, Any]]:
    """
    Extract page-aware text from a PDF or TXT document.

    PDF:
        one record per readable page

    TXT:
        treated as a single page
    """

    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    pages: list[dict[str, Any]] = []

    if extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    else:
        try:
            text = file_bytes.decode("utf-8")

        except UnicodeDecodeError as exc:
            raise ValueError("TXT file must use UTF-8 encoding.") from exc

        text = _clean_text(text)

        if text:
            pages.append(
                {
                    "page": 1,
                    "text": text,
                }
            )

    if not pages:
        raise ValueError("No readable text was found in the document.")

    return pages


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_document(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Split page text into overlapping chunks.

    Each chunk keeps page and chunk metadata so retrieved
    evidence can be traced back to its source.
    """

    chunks: list[dict[str, Any]] = []

    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    if step <= 0:
        raise RuntimeError("Chunk overlap must be smaller than chunk size.")

    chunk_id = 1

    for page_data in pages:
        page_number = page_data["page"]
        words = page_data["text"].split()

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE_WORDS, len(words))

            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

            if end >= len(words):
                break

            start += step

    if not chunks:
        raise ValueError("Unable to create document chunks.")

    return chunks


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

def _normalize_vector(values: list[float]) -> list[float]:
    """
    Convert an embedding to unit length for cosine similarity.
    """

    magnitude = math.sqrt(sum(value * value for value in values))

    if magnitude == 0:
        return values

    return [value / magnitude for value in values]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate semantic embeddings locally using SentenceTransformers.

    The vectors are normalized so cosine similarity can be
    calculated efficiently using a dot product.

    This avoids depending on external embedding API quotas.
    """

    if not texts:
        return []

    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return vectors.tolist()


# ---------------------------------------------------------
# Document Indexing
# ---------------------------------------------------------

def index_document(
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """
    Process and index a document for RAG.

    Pipeline:
        extract
        -> chunk
        -> embed
        -> store vector index
    """

    pages = extract_document_pages(
        filename=filename,
        file_bytes=file_bytes,
    )

    chunks = chunk_document(pages)

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    document_id = str(uuid4())

    indexed_chunks: list[dict[str, Any]] = []

    for chunk, embedding in zip(chunks, embeddings):
        indexed_chunks.append(
            {
                **chunk,
                "embedding": embedding,
            }
        )

    total_characters = sum(
        len(page["text"])
        for page in pages
    )

    RAG_STORE[document_id] = {
        "filename": filename,
        "pages": len(pages),
        "characters": total_characters,
        "chunks": indexed_chunks,
    }

    return {
        "message": "Document indexed successfully.",
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(indexed_chunks),
        "characters": total_characters,
        "embedding_model": EMBEDDING_MODEL,
    }


# ---------------------------------------------------------
# Query Rewriting
# ---------------------------------------------------------

def rewrite_query(question: str) -> str:
    """
    Rewrite a user question into a concise retrieval query.

    If rewriting fails, the original question is used.
    """

    prompt = f"""
You are a retrieval query rewriter.

Rewrite the user's question into one concise standalone search query
that will retrieve the most relevant passages from a document.

Rules:
- Preserve the user's original meaning.
- Do not answer the question.
- Do not add unsupported facts.
- Return only the rewritten query.
- Keep it short and specific.

USER QUESTION:
{question}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )

        rewritten = (response.text or "").strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return question


# ---------------------------------------------------------
# Retrieval Helpers
# ---------------------------------------------------------

def _dot_product(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


def _tokenize(text: str) -> set[str]:
    """
    Create a simple lexical token set for hybrid retrieval.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.casefold(),
        )
    )


def _lexical_score(
    query_tokens: set[str],
    chunk_text: str,
) -> float:
    if not query_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk_text)

    if not chunk_tokens:
        return 0.0

    overlap = query_tokens.intersection(chunk_tokens)

    return len(overlap) / len(query_tokens)


# ---------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------

def retrieve_chunks(
    document_id: str,
    question: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Advanced retrieval pipeline.

    1. Rewrite the query.
    2. Embed original + rewritten query.
    3. Semantic vector retrieval.
    4. Lexical matching.
    5. Hybrid scoring.
    6. Candidate selection.
    7. MMR reranking for relevance + diversity.
    """

    document = RAG_STORE.get(document_id)

    if document is None:
        raise KeyError(
            "Document not found. Upload the document first."
        )

    rewritten_query = rewrite_query(question)

    retrieval_queries = [question]

    if rewritten_query.casefold() != question.casefold():
        retrieval_queries.append(rewritten_query)

    query_vectors = embed_texts(retrieval_queries)

    combined_query_tokens = _tokenize(
        " ".join(retrieval_queries)
    )

    scored_chunks: list[dict[str, Any]] = []

    for chunk in document["chunks"]:
        semantic_score = max(
            _dot_product(
                query_vector,
                chunk["embedding"],
            )
            for query_vector in query_vectors
        )

        lexical_score = _lexical_score(
            combined_query_tokens,
            chunk["text"],
        )

        hybrid_score = (
            SEMANTIC_WEIGHT * semantic_score
            + LEXICAL_WEIGHT * lexical_score
        )

        scored_chunks.append(
            {
                **chunk,
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "retrieval_score": hybrid_score,
            }
        )

    scored_chunks.sort(
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )

    candidates = scored_chunks[:CANDIDATE_K]

    candidates = [
        chunk
        for chunk in candidates
        if chunk["retrieval_score"] >= MIN_RETRIEVAL_SCORE
    ]

    if not candidates:
        return rewritten_query, []

    selected = _mmr_rerank(
        candidates=candidates,
        final_k=FINAL_K,
    )

    return rewritten_query, selected


# ---------------------------------------------------------
# MMR Reranking
# ---------------------------------------------------------

def _mmr_rerank(
    candidates: list[dict[str, Any]],
    final_k: int,
) -> list[dict[str, Any]]:
    """
    Maximum Marginal Relevance reranking.

    MMR balances:
    - relevance to the question
    - diversity between retrieved chunks

    This reduces duplicate overlapping context.
    """

    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    remaining = candidates.copy()

    selected.append(remaining.pop(0))

    while remaining and len(selected) < final_k:
        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:
            max_similarity_to_selected = max(
                _dot_product(
                    candidate["embedding"],
                    selected_chunk["embedding"],
                )
                for selected_chunk in selected
            )

            mmr_score = (
                MMR_LAMBDA * candidate["retrieval_score"]
                - (1 - MMR_LAMBDA)
                * max_similarity_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        if best_candidate is None:
            break

        best_candidate["mmr_score"] = best_score

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


# ---------------------------------------------------------
# Grounded Answer Generation
# ---------------------------------------------------------

def answer_question(
    document_id: str,
    question: str,
) -> dict[str, Any]:
    """
    Retrieve relevant evidence and generate a grounded answer.
    """

    document = RAG_STORE.get(document_id)

    if document is None:
        raise KeyError(
            "Document not found. Upload the document first."
        )

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    rewritten_query, retrieved_chunks = retrieve_chunks(
        document_id=document_id,
        question=question,
    )

    if not retrieved_chunks:
        return {
            "document_id": document_id,
            "filename": document["filename"],
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": "I could not find that information in the document.",
            "sources": [],
            "retrieved_chunks": 0,
        }

    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        context_parts.append(
            (
                f"[Source {source_number} | "
                f"{document['filename']} | "
                f"page {chunk['page']} | "
                f"chunk {chunk['chunk_id']}]\n"
                f"{chunk['text']}"
            )
        )

        sources.append(
            {
                "source": source_number,
                "filename": document["filename"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "retrieval_score": round(
                    chunk["retrieval_score"],
                    4,
                ),
            }
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a grounded document question-answering assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not supported by the retrieved context, say:
  "I could not find that information in the document."
- Cite supporting evidence using [Source 1], [Source 2], etc.
- Keep the answer clear and directly relevant to the question.

ORIGINAL QUESTION:
{question}

RETRIEVAL QUERY:
{rewritten_query}

RETRIEVED CONTEXT:
{context}
"""

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(retrieved_chunks),
    }
