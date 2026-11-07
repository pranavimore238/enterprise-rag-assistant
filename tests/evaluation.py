
import streamlit as st
import pandas as pd
import chromadb

from tests.test_rag import (
    evaluate_retrieval,
    evaluate_answer
)


st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Evaluation Dashboard")
st.write("Evaluate retrieval and answer-generation performance.")

# Connect to ChromaDB
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
        available_documents.setdefault(name, set()).add(document_id)


st.divider()
st.header("Select documents")

if available_documents:

    options = ["All Documents"] + sorted(available_documents.keys())

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
        selected_ids = list(available_documents[selected])

else:
    selected_ids = None
    st.info("No uploaded PDFs found. The original BTU document will be used.")


# Evaluation questions
questions = [
    "What is the educational qualification?",
    "What professional experience is mentioned?",
    "What technical skills are listed?"
]

st.divider()
st.header("Evaluation questions")

for question in questions:
    st.write("•", question)


if st.button("Run RAG Evaluation"):

    results = []

    with st.spinner("Evaluating RAG performance..."):

        for question in questions:

            retrieval = evaluate_retrieval(
                question,
                document_ids=selected_ids
            )

            answer_result = evaluate_answer(
                question,
                document_ids=selected_ids
            )

            results.append({
                "Question": question,
                "Retrieved Chunks": retrieval["retrieved_chunks"],
                "Retrieval Time (s)": retrieval["retrieval_time_seconds"],
                "Response Time (s)": answer_result["response_time_seconds"],
                "Source Count": answer_result["source_count"],
                "Answer": answer_result["answer"]
            })

    st.session_state["evaluation_results"] = results


# Display evaluation results
if "evaluation_results" in st.session_state:

    results = st.session_state["evaluation_results"]

    st.divider()
    st.header("Evaluation Results")

    average_retrieval = sum(
        item["Retrieval Time (s)"] for item in results
    ) / len(results)

    average_response = sum(
        item["Response Time (s)"] for item in results
    ) / len(results)

    col1, col2, col3 = st.columns(3)

    col1.metric("Questions Evaluated", len(results))
    col2.metric("Average Retrieval Time", f"{average_retrieval:.3f} s")
    col3.metric("Average Response Time", f"{average_response:.2f} s")

    st.subheader("Performance Table")

    dataframe = pd.DataFrame(results)

    st.dataframe(
        dataframe.drop(columns=["Answer"]),
        use_container_width=True
    )

    st.subheader("Generated Answers")

    for item in results:
        with st.expander(item["Question"]):
            st.write(item["Answer"])

    csv = dataframe.to_csv(index=False)

    st.download_button(
        "Download Evaluation Results",
        data=csv,
        file_name="rag_evaluation_results.csv",
        mime="text/csv"
    )
