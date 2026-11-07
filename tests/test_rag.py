

import time
import json
import re

from src.rag_pipeline import (
    retrieve_relevant_chunks,
    generate_answer,
    llm
)


# --------------------------------------------------
# 1. RETRIEVAL EVALUATION
# --------------------------------------------------

def evaluate_retrieval(question, document_ids=None):
    """Measure document retrieval performance."""

    start_time = time.time()

    chunks = retrieve_relevant_chunks(
        question,
        document_ids=document_ids
    )

    retrieval_time = time.time() - start_time

    return {
        "question": question,
        "retrieved_chunks": len(chunks),
        "retrieval_time_seconds": round(retrieval_time, 3),
        "retrieved_documents": list(set(
            chunk["document_name"] for chunk in chunks
        )),
        "sources": chunks
    }


# --------------------------------------------------
# 2. ANSWER GENERATION EVALUATION
# --------------------------------------------------

def evaluate_answer(question, document_ids=None):
    """Measure answer-generation performance."""

    start_time = time.time()

    answer, sources = generate_answer(
        question,
        document_ids=document_ids
    )

    response_time = time.time() - start_time

    return {
        "question": question,
        "answer": answer,
        "response_time_seconds": round(response_time, 3),
        "source_count": len(sources),
        "sources": sources
    }


# --------------------------------------------------
# 3. HEURISTIC ANSWER QUALITY EVALUATION
# --------------------------------------------------

def evaluate_answer_quality(answer, sources):
    """Estimate answer quality using text-based checks."""

    if not answer or not sources:
        return {
            "faithfulness_score": 0,
            "relevance_score": 0,
            "completeness_score": 0,
            "overall_score": 0
        }

    answer_words = {
        word.strip(".,!?():;")
        for word in answer.lower().split()
        if len(word.strip(".,!?():;")) > 3
    }

    source_text = " ".join(
        chunk["text"] for chunk in sources
    ).lower()

    source_words = set(source_text.split())

    supported_words = answer_words.intersection(source_words)

    faithfulness = (
        len(supported_words) / len(answer_words)
        if answer_words else 0
    )

    relevance = min(len(answer.split()) / 50, 1)

    completeness = min(len(answer.split()) / 80, 1)

    faithfulness_score = round(faithfulness * 100, 2)
    relevance_score = round(relevance * 100, 2)
    completeness_score = round(completeness * 100, 2)

    overall_score = round(
        (
            faithfulness_score
            + relevance_score
            + completeness_score
        ) / 3,
        2
    )

    return {
        "faithfulness_score": faithfulness_score,
        "relevance_score": relevance_score,
        "completeness_score": completeness_score,
        "overall_score": overall_score
    }


# --------------------------------------------------
# 4. LLM-AS-A-JUDGE EVALUATION
# --------------------------------------------------

def evaluate_with_llm_judge(question, answer, sources):
    """
    Use the local Qwen model to evaluate generated answers
    against retrieved PDF context.
    """

    if not answer or not sources:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "overall_score": 0,
            "reason": "Missing answer or source documents."
        }

    context = "\n\n".join(
        chunk["text"] for chunk in sources
    )

    prompt = f"""
You are an independent evaluator of a
Retrieval-Augmented Generation system.

Evaluate the generated answer using only the
supplied retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context[:7000]}

GENERATED ANSWER:
{answer}

Give three scores from 1 to 5.

1. Faithfulness:
Is the answer supported by the retrieved context?

2. Relevance:
Does the answer directly address the question?

3. Completeness:
Does the answer cover the important information?

SCORING:
1 = Very poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

If information is missing from the context,
do not reward unsupported claims.

Return ONLY valid JSON in this format:

{{
    "faithfulness": 1,
    "relevance": 1,
    "completeness": 1,
    "reason": "Brief explanation of the evaluation"
}}
"""

    try:
        response = llm.invoke(prompt)

        response_text = response.content.strip()

        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "The judge did not return valid JSON."
            )

        scores = json.loads(match.group())

        def validate_score(value):
            if isinstance(value, bool):
                raise ValueError("Invalid score type.")

            score = float(value)

            if not score.is_integer() or not 1 <= score <= 5:
                raise ValueError("Score must be an integer from 1 to 5.")

            return int(score)

        faithfulness = validate_score(
            scores["faithfulness"]
        )

        relevance = validate_score(
            scores["relevance"]
        )

        completeness = validate_score(
            scores["completeness"]
        )

        overall_score = round(
            (
                faithfulness
                + relevance
                + completeness
            ) / 3,
            2
        )

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "overall_score": overall_score,
            "reason": str(
                scores.get(
                    "reason",
                    "No explanation provided."
                )
            )
        }

    except Exception as error:

        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "overall_score": 0,
            "reason": f"Judge evaluation failed: {error}"
        }
