import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pytesseract
from django.conf import settings
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Document, DocumentChunk

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 200
HYBRID_RETRIEVAL_THRESHOLD = 0.15
SYSTEM_PROMPT = (
    "Answer using ONLY the context below. If the answer is not in the context, reply exactly: "
    "I couldn't find this in the uploaded documents. Quote numbers exactly as written. "
    "Cite document name and page for every claim."
)


def _configure_tesseract() -> None:
    env_path = os.getenv("TESSERACT_CMD")
    if env_path:
        pytesseract.pytesseract.tesseract_cmd = env_path
        return

    resolved = shutil.which("tesseract")
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved
        return

    candidates = [
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            logger.info("Configured OCR path: %s", candidate)
            return


_configure_tesseract()


def compute_file_hash(file_obj) -> str:
    if hasattr(file_obj, "read"):
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        contents = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        contents = file_obj
    return hashlib.sha256(contents).hexdigest()


def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r", "\n")
    lines = []
    seen = set()
    for line in text.split("\n"):
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" -_|#*=")
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", cleaned):
            continue
        if re.fullmatch(r"(page|Page)\s*\d+", cleaned):
            continue
        if re.fullmatch(r"[\-_=#*]+", cleaned):
            continue
        if re.fullmatch(r"(?i)(rice grain detection|project report|document summary)", cleaned):
            continue
        if len(cleaned) < 3 and not any(ch.isalpha() for ch in cleaned):
            continue
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(cleaned)

    return "\n".join(lines)


def _strip_cover_boilerplate(page_text: str) -> str:
    cleaned = page_text or ""
    cleaned = re.sub(r"(?im)^\s*(?:Rice Grain Detection\s*-\s*Project Report|Project Report)\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*Page\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:Date|Prepared|Generated)\s*[:\-]?\s*.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:https?://\S+|www\.\S+)\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ocr_page_text(page) -> str:
    try:
        import fitz
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("OCR fallback unavailable: %s", exc)
        return ""

    try:
        _configure_tesseract()
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(image)
        return clean_extracted_text(_strip_cover_boilerplate(text))
    except Exception as exc:  # pragma: no cover - runtime OCR failure
        logger.warning("OCR failed for a PDF page: %s", exc)
        return ""


def extract_text_from_pdf(file_obj) -> list[dict]:
    import fitz

    documents = []
    with fitz.open(stream=file_obj.read(), filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_extracted_text(_strip_cover_boilerplate(page.get_text("text") or ""))
            if len(text) < 200:
                ocr_text = _ocr_page_text(page)
                if ocr_text and len(ocr_text) > len(text):
                    logger.warning("OCR fallback used for PDF page %s", page_number)
                    text = ocr_text
            documents.append({"page_number": page_number, "text": text})
    return documents


def extract_text_from_docx(file_obj) -> list[dict]:
    from docx import Document as DocxDocument

    doc = DocxDocument(file_obj)
    paragraphs = []
    for paragraph in doc.paragraphs:
        cleaned = clean_extracted_text(_strip_cover_boilerplate(paragraph.text))
        if cleaned:
            paragraphs.append(cleaned)
    text = "\n".join(paragraphs)
    return [{"page_number": 1, "text": text}]


def extract_text_from_txt(file_obj) -> list[dict]:
    content = clean_extracted_text(_strip_cover_boilerplate(file_obj.read().decode("utf-8", errors="replace")))
    return [{"page_number": 1, "text": content}]


def chunk_page_text(page_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, min_length: int = MIN_CHUNK_LENGTH) -> List[str]:
    cleaned = re.sub(r"\s+", " ", page_text or "").strip()
    if not cleaned:
        return []

    cleaned = cleaned.replace("\u00a0", " ")
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            last_space = cleaned.rfind(" ", start + max(1, chunk_size - overlap), end)
            if last_space > start + min_length:
                end = last_space
        chunk = cleaned[start:end].strip()
        if chunk:
            chunk = re.sub(r"(?i)\b(page\s+\d+\s+)?\d+\.\s*$", "", chunk)
            chunk = re.sub(r"(?i)^\b(page\s+\d+\s+)?\d+\.\s*", "", chunk)
            chunk = re.sub(r"\s{2,}", " ", chunk).strip()
        if len(chunk) >= min_length or not chunks:
            if chunk:
                chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


def get_embedding_model() -> SentenceTransformer:
    if not hasattr(get_embedding_model, "_model"):
        get_embedding_model._model = SentenceTransformer(MODEL_NAME)
    return get_embedding_model._model


def get_cross_encoder_model():
    if not hasattr(get_cross_encoder_model, "_model"):
        get_cross_encoder_model._model = CrossEncoder(CROSS_ENCODER_MODEL)
    return get_cross_encoder_model._model


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def _chunk_document_id(chunk):
    document = getattr(chunk, "document", None)
    if hasattr(chunk, "document_id"):
        return getattr(chunk, "document_id")
    if document is not None:
        return getattr(document, "id", None)
    return None


def deduplicate_chunk_scores(scores):
    best_by_page = {}
    for chunk, score in scores:
        key = (_chunk_document_id(chunk), chunk.page_number)
        current = best_by_page.get(key)
        if current is None or score > current[1]:
            best_by_page[key] = (chunk, score)
    return sorted(best_by_page.values(), key=lambda item: item[1], reverse=True)


def deduplicate_chunks(chunks):
    best_by_page = {}
    for chunk in chunks:
        key = (_chunk_document_id(chunk), chunk.page_number)
        if key not in best_by_page:
            best_by_page[key] = chunk
    return list(best_by_page.values())


def index_document(document: Document) -> int:
    file_name = document.file.name.lower()
    file_obj = document.file.open("rb")
    document.file_hash = compute_file_hash(file_obj)
    file_obj.close()

    if Document.objects.filter(file_hash=document.file_hash).exclude(pk=document.pk).exists():
        logger.warning("Skipping duplicate document index for %s (hash=%s)", document.title, document.file_hash)
        document.status = "indexed"
        document.save(update_fields=["file_hash", "status", "updated_at"])
        return 0

    document.save(update_fields=["file_hash"])

    file_obj = document.file.open("rb")
    try:
        if file_name.endswith(".pdf"):
            pages = extract_text_from_pdf(file_obj)
        elif file_name.endswith(".docx"):
            pages = extract_text_from_docx(file_obj)
        elif file_name.endswith(".txt"):
            pages = extract_text_from_txt(file_obj)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
    finally:
        file_obj.close()

    document.page_count = max(len(pages), 1)
    document.status = "processing"
    document.save(update_fields=["page_count", "status"])

    chunks_to_create = []
    for page in pages:
        page_text = page.get("text", "")
        if not page_text.strip():
            continue
        for i, chunk_text_value in enumerate(chunk_page_text(page_text)):
            chunks_to_create.append(
                DocumentChunk(
                    document=document,
                    text=chunk_text_value,
                    page_number=page.get("page_number", 1),
                    chunk_index=i,
                    embedding=[],
                )
            )

    if not chunks_to_create:
        document.status = "indexed"
        document.save(update_fields=["status", "updated_at"])
        return 0

    embeddings = embed_texts([chunk.text for chunk in chunks_to_create])
    for chunk, embedding in zip(chunks_to_create, embeddings):
        chunk.embedding = embedding.tolist()

    DocumentChunk.objects.filter(document=document).delete()
    DocumentChunk.objects.bulk_create(chunks_to_create)

    document.status = "indexed"
    document.save(update_fields=["status", "updated_at"])
    return len(chunks_to_create)


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def find_best_answer_snippet(question: str, chunks) -> str:
    if not chunks:
        return ""

    question_lower = question.lower()
    question_terms = set(re.findall(r"[a-zA-Z0-9]+", question_lower))
    best_sentence = ""
    best_score = -1

    for chunk in chunks:
        for sentence in split_into_sentences(chunk.text):
            sentence_lower = sentence.lower()
            sentence_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence_lower))
            overlap = len(question_terms & sentence_terms)

            if any(term in question_lower for term in ["college", "school", "university", "studied", "study"]):
                if any(term in sentence_lower for term in ["college", "school", "university", "studied", "study", "graduated"]):
                    overlap += 5

            if any(term in question_lower for term in ["course", "degree", "program", "branch", "engineering", "btech"]):
                if any(term in sentence_lower for term in ["b.tech", "computer science", "engineering", "ai & ds", "ai", "ds"]):
                    overlap += 6

            if any(term in question_lower for term in ["project", "projects", "done", "developed", "built"]):
                if any(term in sentence_lower for term in ["project", "projects", "developed", "built", "model", "application"]):
                    overlap += 6

            if "poojitha" in sentence_lower or "veni" in sentence_lower:
                overlap += 1
            if sentence_lower in question_lower:
                overlap += 2

            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

    if best_sentence:
        return best_sentence

    first_chunk = chunks[0].text
    sentences = split_into_sentences(first_chunk)
    return sentences[0] if sentences else first_chunk[:220]


def build_citation_sources(chunks):
    deduped = {}
    for chunk in chunks:
        key = (_chunk_document_id(chunk), chunk.page_number)
        if key in deduped:
            continue
        snippet = find_best_answer_snippet("", [chunk]) if chunk.text else ""
        deduped[key] = {
            "document": chunk.document.title,
            "page": chunk.page_number,
            "snippet": snippet[:220] if snippet else chunk.text[:220],
            "document_id": getattr(chunk.document, "id", _chunk_document_id(chunk)),
        }
    return list(deduped.values())


def _hybrid_chunk_scores(question: str, chunks) -> List[tuple]:
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    question_embedding = get_embedding_model().encode(question, convert_to_numpy=True, show_progress_bar=False)
    embedding_matrix = embed_texts(texts)
    embedding_scores = cosine_similarity(question_embedding.reshape(1, -1), embedding_matrix).ravel()

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = tfidf.fit_transform(texts + [question])
    keyword_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).ravel()

    emb_min = float(np.min(embedding_scores)) if len(embedding_scores) else 0.0
    emb_max = float(np.max(embedding_scores)) if len(embedding_scores) else 0.0
    emb_range = max(emb_max - emb_min, 1e-9)
    key_min = float(np.min(keyword_scores)) if len(keyword_scores) else 0.0
    key_max = float(np.max(keyword_scores)) if len(keyword_scores) else 0.0
    key_range = max(key_max - key_min, 1e-9)

    hybrid_scores = []
    for idx, chunk in enumerate(chunks):
        emb_score = (embedding_scores[idx] - emb_min) / emb_range
        key_score = (keyword_scores[idx] - key_min) / key_range
        hybrid = 0.6 * emb_score + 0.4 * key_score
        hybrid_scores.append((chunk, float(hybrid)))

    return sorted(hybrid_scores, key=lambda item: item[1], reverse=True)


def _rerank_with_cross_encoder(question: str, chunks) -> List[tuple]:
    if not chunks:
        return []

    try:
        cross_encoder = get_cross_encoder_model()
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = cross_encoder.predict(pairs)
        return sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    except Exception as exc:  # pragma: no cover - model download or runtime issue
        logger.warning("Cross-encoder rerank failed: %s", exc)
        return sorted(chunks, key=lambda chunk: chunk.text.lower().count(question.lower().split()[0]) if question.lower().split() else 0, reverse=True)


def retrieve_relevant_chunks(question: str, document_ids: Optional[List[int]] = None, top_k: int = 5, fetch_k: int = 15):
    queryset = DocumentChunk.objects.select_related("document")
    if document_ids:
        queryset = queryset.filter(document_id__in=document_ids)

    chunks = list(queryset)
    if not chunks:
        return []

    initial_scored = _hybrid_chunk_scores(question, chunks)
    ranked = deduplicate_chunk_scores(initial_scored)
    if not ranked:
        return []

    candidates = [chunk for chunk, _ in ranked[: max(15, fetch_k)]]
    reranked = _rerank_with_cross_encoder(question, candidates)
    filtered = [(chunk, score) for chunk, score in reranked if float(score) >= HYBRID_RETRIEVAL_THRESHOLD]
    if not filtered:
        return []
    return [chunk for chunk, _ in filtered[: min(top_k, len(filtered))]]


def is_comparison_question(question: str) -> bool:
    question_lower = (question or "").lower()
    comparison_words = ["both", "compare", "difference", "which document", "documents both", "do rice and", "do food", "which document mentions"]
    return any(word in question_lower for word in comparison_words) or ("and" in question_lower and "document" in question_lower)


def retrieve_document_chunks_for_comparison(question: str, document_ids: Optional[List[int]] = None, top_k: int = 1):
    queryset = Document.objects.all()
    if document_ids:
        queryset = queryset.filter(id__in=document_ids)

    results = []
    for document in queryset:
        doc_chunks = list(DocumentChunk.objects.select_related("document").filter(document_id=document.id))
        if not doc_chunks:
            continue

        ranked = deduplicate_chunk_scores(_hybrid_chunk_scores(question, doc_chunks))
        if not ranked:
            continue

        results.extend(chunk for chunk, _ in ranked[: max(1, top_k)])

    return deduplicate_chunks(results)


def _mask_api_key(value: str) -> str:
    if not value:
        return "<empty>"
    value = value.strip()
    if len(value) <= 4:
        return f"{value[:1]}****"
    return f"{value[:4]}****"


def _sanitize_llm_error(message: str) -> str:
    if not message:
        return ""
    sanitized = re.sub(r"sk-[A-Za-z0-9_]+", "sk-****", message)
    sanitized = re.sub(r"(?i)(openai|anthropic)[_-]?api[-_ ]?key[:= ]+[A-Za-z0-9._-]+", r"\1 API key=<redacted>", sanitized)
    return sanitized


def get_llm_provider():
    anthropic_key = (getattr(settings, "ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    openai_key = (getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()
    model_name = (getattr(settings, "LLM_MODEL", "") or os.getenv("LLM_MODEL", "")).strip()

    if anthropic_key:
        return "anthropic", anthropic_key, model_name or "claude-haiku-5-5"
    if openai_key:
        return "openai", openai_key, model_name or "gpt-4o-mini"
    return None, "", ""


def _build_context_text(context_chunks) -> str:
    if not context_chunks:
        return ""
    return "\n\n".join(
        f"[doc: {chunk.document.title}, page {chunk.page_number}]\n{chunk.text}"
        for chunk in context_chunks
    )


def generate_grounded_answer(question: str, context_chunks, conversation_history=None):
    if not context_chunks:
        logger.warning("No retrieval context for question: %s", question)
        return "I couldn't find this in the uploaded documents."

    llm_context = deduplicate_chunks(context_chunks)[:5]
    if not llm_context:
        logger.warning("No usable LLM context for question: %s", question)
        return "I couldn't find this in the uploaded documents."

    context_text = _build_context_text(llm_context)
    provider, api_key, model = get_llm_provider()
    if not provider:
        fallback = find_best_answer_snippet(question, llm_context) or "I couldn't find this in the uploaded documents."
        logger.warning("No LLM API key configured; using extractive fallback for question: %s", question)
        return f"Answered by fallback\n\nThe uploaded documents support this answer: {fallback}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in (conversation_history or []):
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"})

    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=500,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Question: {question}\n\nContext:\n{context_text}"}],
            )
            answer = response.content[0].text.strip()
            logger.info("Grounded answer generated with Anthropic for question: %s using model %s", question, model)
            return f"Answered by LLM ({model})\n\n{answer}"

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("Grounded answer generated with OpenAI for question: %s using model %s", question, model)
        return f"Answered by LLM ({model})\n\n{answer}"
    except Exception as exc:
        error_message = _sanitize_llm_error(str(exc))
        logger.error(
            "LLM grounded-answer call failed for question %s with %s: %s",
            question,
            type(exc).__name__,
            error_message,
        )
        fallback = find_best_answer_snippet(question, llm_context) or "I couldn't find this in the uploaded documents."
        return (
            f"LLM call failed: {type(exc).__name__}: {error_message}; using fallback.\n\n"
            f"Answered by fallback\n\n{fallback}"
        )
