
import streamlit as st
import pandas as pd
import chromadb

from tests.test_rag import (
    evaluate_retrieval,
    evaluate_answer,
    evaluate_answer_quality,
    evaluate_with_llm_judge
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise RAG Evaluation Dashboard")

st.write(
    "Evaluate retrieval performance, answer generation, "
    "and answer quality using a local LLM judge."
)


# --------------------------------------------------
# DOCUMENT SELECTION
# --------------------------------------------------

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="uploaded_documents"
)

records = collection.get(include=["metadatas"])

available_documents = {}

for metadata in records["metadatas"]:

    name = metadata.get("document_name")
    document_id = metadata.get("document_id")

    if name and document_id:

        available_documents.setdefault(
            name, set()
        ).add(document_id)


st.divider()

st.header("Select documents")

if available_documents:

    options = ["All Documents"] + sorted(
        available_documents.keys()
    )

    selected = st.selectbox(
        "Choose documents for evaluation:",
        options
    )

    if selected == "All Documents":

        selected_ids = list({
            doc_id
            for ids in available_documents.values()
            for doc_id in ids
        })

    else:

        selected_ids = list(
            available_documents[selected]
        )

else:

    selected_ids = None

    st.info(
        "No uploaded PDFs found. "
        "The original BTU document will be used."
    )


# --------------------------------------------------
# EVALUATION QUESTIONS
# --------------------------------------------------

questions = [
    "What is the educational qualification?",
    "What professional experience is mentioned?",
    "What technical skills are listed?"
]


# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------

st.divider()

st.header("Run Evaluation")

if st.button(
    "Run Complete RAG Evaluation",
    type="primary"
):

    results = []

    progress = st.progress(0)

    status = st.empty()

    with st.spinner(
        "Running RAG evaluation and LLM judge..."
    ):

        for index, question in enumerate(questions):

            status.write(
                f"Evaluating question {index + 1} "
                f"of {len(questions)}..."
            )

            # Retrieval evaluation
            retrieval = evaluate_retrieval(
                question,
                document_ids=selected_ids
            )

            # Answer generation
            answer_result = evaluate_answer(
                question,
                document_ids=selected_ids
            )

            # Preliminary heuristic evaluation
            quality = evaluate_answer_quality(
                answer_result["answer"],
                answer_result["sources"]
            )

            # LLM judge evaluation
            judge = evaluate_with_llm_judge(
                question,
                answer_result["answer"],
                answer_result["sources"]
            )

            results.append({

                "Question": question,

                "Retrieved Chunks":
                    retrieval["retrieved_chunks"],

                "Retrieval Time (s)":
                    retrieval["retrieval_time_seconds"],

                "Response Time (s)":
                    answer_result["response_time_seconds"],

                "Source Count":
                    answer_result["source_count"],

                "Heuristic Faithfulness (%)":
                    quality["faithfulness_score"],

                "Heuristic Relevance (%)":
                    quality["relevance_score"],

                "Heuristic Completeness (%)":
                    quality["completeness_score"],

                "LLM Faithfulness (/5)":
                    judge["faithfulness"],

                "LLM Relevance (/5)":
                    judge["relevance"],

                "LLM Completeness (/5)":
                    judge["completeness"],

                "LLM Overall (/5)":
                    judge["overall_score"],

                "Judge Explanation":
                    judge["reason"],

                "Answer":
                    answer_result["answer"]

            })

            progress.progress(
                (index + 1) / len(questions)
            )

    status.success("Evaluation completed!")

    st.session_state["evaluation_results"] = results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "evaluation_results" in st.session_state:

    results = st.session_state["evaluation_results"]

    st.divider()

    st.header("Evaluation Results")

    # Performance metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Questions",
        len(results)
    )

    col2.metric(
        "Average Retrieval Time",
        f"{sum(r['Retrieval Time (s)'] for r in results) / len(results):.2f} sec"
    )

    col3.metric(
        "Average Response Time",
        f"{sum(r['Response Time (s)'] for r in results) / len(results):.2f} sec"
    )

    # LLM judge scores
    st.divider()

    st.subheader("LLM-as-a-Judge Scores")

    valid_results = [
        r for r in results
        if r["LLM Overall (/5)"] > 0
    ]

    if valid_results:

        faithfulness_avg = sum(
            r["LLM Faithfulness (/5)"]
            for r in valid_results
        ) / len(valid_results)

        relevance_avg = sum(
            r["LLM Relevance (/5)"]
            for r in valid_results
        ) / len(valid_results)

        completeness_avg = sum(
            r["LLM Completeness (/5)"]
            for r in valid_results
        ) / len(valid_results)

        overall_avg = sum(
            r["LLM Overall (/5)"]
            for r in valid_results
        ) / len(valid_results)

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Faithfulness",
            f"{faithfulness_avg:.2f}/5"
        )

        q2.metric(
            "Relevance",
            f"{relevance_avg:.2f}/5"
        )

        q3.metric(
            "Completeness",
            f"{completeness_avg:.2f}/5"
        )

        q4.metric(
            "Overall Score",
            f"{overall_avg:.2f}/5"
        )

    else:

        st.warning(
            "The LLM judge could not produce valid scores. "
            "Check the judge explanations below."
        )

    st.caption(
        "LLM judge scores are model-generated assessments, "
        "not independently verified ground truth."
    )

    # Detailed table
    st.divider()

    st.subheader("Detailed Performance Metrics")

    dataframe = pd.DataFrame(results)

    st.dataframe(
        dataframe.drop(
            columns=["Answer", "Judge Explanation"]
        ),
        use_container_width=True
    )

    # Answers and judge explanations
    st.divider()

    st.subheader("Generated Answers and Judge Feedback")

    for index, result in enumerate(results, start=1):

        with st.expander(
            f"Question {index}: {result['Question']}"
        ):

            st.markdown("**Generated Answer**")

            st.write(result["Answer"])

            st.markdown("**LLM Judge Scores**")

            st.write(
                f"Faithfulness: {result['LLM Faithfulness (/5)']}/5"
            )

            st.write(
                f"Relevance: {result['LLM Relevance (/5)']}/5"
            )

            st.write(
                f"Completeness: {result['LLM Completeness (/5)']}/5"
            )

            st.write(
                f"Overall: {result['LLM Overall (/5)']}/5"
            )

            st.markdown("**Judge Explanation**")

            st.write(result["Judge Explanation"])

    # CSV download
    csv_data = dataframe.to_csv(index=False)

    st.download_button(
        label="Download Complete Evaluation Results",
        data=csv_data,
        file_name="enterprise_rag_evaluation.csv",
        mime="text/csv"
    )
