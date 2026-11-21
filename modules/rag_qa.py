"""Evidence-grounded question answering for the EDA AI Agent.

Exact numeric and categorical statistics already calculated by Python are
returned directly. Other questions use the local LLM with retrieved evidence.
"""

import re

from modules.rag_engine import EDARetrievalEngine
from modules.llm_engine import generate_response


RAG_SYSTEM_PROMPT = """
You are the question-answering assistant of the EDA AI Agent.
Answer using ONLY the retrieved dataset evidence.
- Answer the user's exact question.
- Never invent statistics, values, or column information.
- Do not combine values from different columns.
- If evidence is insufficient, say what is missing.
- Explain statistical findings in simple language.
- Correlation does not establish causation.
Keep the answer concise.
"""


def _normalise_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _question_mentions_column(question, column):
    question_words = set(_normalise_text(question).split())
    column_words = _normalise_text(column).split()
    return bool(column_words) and all(word in question_words for word in column_words)


def _direct_categorical_answer(question, documents):
    """Return a retrieved categorical distribution when its column is named."""
    question_text = _normalise_text(question)
    candidates = []

    for document in documents:
        text = str(document)
        match = re.search(
            r"Categorical distribution for column '([^']+)':\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match and _question_mentions_column(question_text, match.group(1)):
            candidates.append((match.group(1), text))

    if len(candidates) == 1:
        column, evidence = candidates[0]
        return f"Here is the observed distribution for '{column}':\n{evidence}"

    # A general distribution question can use one unambiguous retrieved result.
    if not candidates and len(documents) == 1:
        text = str(documents[0])
        if text.lower().startswith("categorical distribution for column '"):
            return text

    return None


def _requested_numeric_metric(question):
    q = _normalise_text(question)

    if any(term in q.split() for term in ("average", "mean")):
        return "mean", "average"
    if "median" in q.split():
        return "median", "median"
    if "standard deviation" in q:
        return "standard deviation", "standard deviation"
    if "variance" in q.split():
        return "variance", "variance"
    if "skewness" in q.split() or "skew" in q.split():
        return "skewness", "skewness"
    return None, None


def _direct_numeric_answer(question, documents):
    """Extract a requested metric from a Python-generated numeric finding."""
    metric, display_name = _requested_numeric_metric(question)
    if metric is None:
        return None

    candidates = []
    for document in documents:
        text = str(document)
        match = re.search(
            r"(?:Numeric distribution for column|Numeric summary for column)"
            r"\s*'([^']+)'\s*:\s*(.*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            # Also support the older insight format: 'Age': n=..., mean=...
            match = re.search(
                r"['\"]([^'\"]+)['\"]\s*:\s*n\s*=\s*[^,]+,\s*(.*)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if not match:
            continue

        column, statistics = match.group(1), match.group(2)
        if not _question_mentions_column(question, column):
            continue

        metric_pattern = {
            "mean": r"\bmean\s*=\s*([^,;.]+)",
            "median": r"\bmedian\s*=\s*([^,;.]+)",
            "standard deviation": r"\bstandard deviation\s*=\s*([^,;.]+)",
            "variance": r"\bvariance\s*=\s*([^,;.]+)",
            "skewness": r"\bskewness\s*=\s*([^,;.]+)",
        }[metric]
        value_match = re.search(metric_pattern, statistics, flags=re.IGNORECASE)
        if value_match:
            candidates.append((column, value_match.group(1).strip()))

    if len(candidates) == 1:
        column, value = candidates[0]
        unit = " years" if _normalise_text(column) == "age" else ""
        return f"The {display_name} of '{column}' is {value}{unit}."

    return None


def answer_dataset_question(
    question,
    collection_name="eda_uploaded_dataset_knowledge",
    top_k=3,
):
    """Retrieve relevant dataset findings and answer from that evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a valid question.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    rag_engine = EDARetrievalEngine()

    try:
        rag_engine.collection = rag_engine.client.get_collection(
            name=collection_name
        )
    except Exception as error:
        raise ValueError(
            f"Could not open ChromaDB collection '{collection_name}': {error}"
        ) from error

    retrieved_documents = rag_engine.retrieve_documents(
        question=question,
        top_k=top_k,
    )

    if not retrieved_documents:
        return {
            "status": "no_evidence",
            "question": question,
            "retrieved_evidence": [],
            "answer": "I could not find relevant evidence in the indexed dataset information.",
            "llm_model": None,
            "evidence_source": "ChromaDB retrieval",
        }

    # Exact statistics are taken directly from Python-generated evidence.
    direct_answer = _direct_numeric_answer(question, retrieved_documents)
    if direct_answer is None:
        direct_answer = _direct_categorical_answer(question, retrieved_documents)

    if direct_answer is not None:
        return {
            "status": "success",
            "question": question,
            "retrieved_evidence": retrieved_documents,
            "answer": direct_answer,
            "llm_model": None,
            "evidence_source": "ChromaDB retrieval; Python-computed statistic",
        }

    evidence = "\n".join(f"- {document}" for document in retrieved_documents)
    prompt = f"""
USER QUESTION:
{question}

RETRIEVED DATASET EVIDENCE:
{evidence}

Answer the exact question using only this evidence.
Do not combine values from different columns.
Do not invent counts, percentages, or statistics.
If the evidence is insufficient, explain what is missing.
"""

    answer = generate_response(
        prompt=prompt,
        system_prompt=RAG_SYSTEM_PROMPT,
        num_predict=220,
    )

    return {
        "status": "success",
        "question": question,
        "retrieved_evidence": retrieved_documents,
        "answer": answer,
        "llm_model": "llama3.2:1b",
        "evidence_source": "ChromaDB dataset retrieval",
    }
