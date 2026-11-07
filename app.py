
import streamlit as st
import chromadb

from src.rag_pipeline import generate_answer
from src.document_loader import load_pdf
from src.embeddings import chunk_documents, create_embeddings
from src.vector_store import save_to_chroma


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise AI Knowledge Assistant")
st.write("Upload PDF documents and ask questions about their content.")
st.divider()


# --------------------------------------------------
# CHROMADB CONNECTION
# --------------------------------------------------

client = chromadb.PersistentClient(path="chroma_db")

uploaded_collection = client.get_or_create_collection(
    name="uploaded_documents"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_document_ids" not in st.session_state:
    st.session_state.uploaded_document_ids = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About this project")

    st.write("**LLM:** Qwen 2.5")
    st.write("**Vector Database:** ChromaDB")
    st.write("**Embeddings:** MiniLM")
    st.write("**Framework:** Streamlit")

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # DOCUMENT MANAGEMENT

    st.header("📂 Document Management")

    records = uploaded_collection.get(
        include=["metadatas"]
    )

    document_info = {}

    for metadata in records["metadatas"]:

        document_name = metadata.get("document_name")

        if document_name:
            document_info[document_name] = (
                document_info.get(document_name, 0) + 1
            )

    if document_info:

        st.write("**Uploaded documents:**")

        for document_name, chunk_count in document_info.items():

            st.write(f"📄 {document_name}")
            st.caption(f"{chunk_count} chunks")

            if st.button(
                f"Delete {document_name}",
                key=f"delete_{document_name}"
            ):

                matching_ids = [
                    record_id
                    for record_id, metadata in zip(
                        records["ids"],
                        records["metadatas"]
                    )
                    if metadata.get("document_name") == document_name
                ]

                uploaded_collection.delete(ids=matching_ids)

                # Update selected document IDs
                remaining_ids = []

                for document_id in st.session_state.uploaded_document_ids:

                    document_records = uploaded_collection.get(
                        where={"document_id": document_id},
                        include=["metadatas"]
                    )

                    if document_records["ids"]:
                        remaining_ids.append(document_id)

                st.session_state.uploaded_document_ids = remaining_ids

                st.success(f"{document_name} deleted successfully!")

                st.rerun()

    else:
        st.info("No uploaded documents available.")


# --------------------------------------------------
# MULTIPLE PDF UPLOAD
# --------------------------------------------------

st.header("📄 Upload your PDF documents")

uploaded_files = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write(f"Selected {len(uploaded_files)} PDF document(s).")

    for file in uploaded_files:
        st.write("📄", file.name)

    if st.button("Process All PDFs"):

        document_ids = []

        with st.spinner("Processing your PDF documents..."):

            try:

                for uploaded_file in uploaded_files:

                    st.write(f"Processing: {uploaded_file.name}")

                    # Extract PDF text
                    uploaded_file.seek(0)
                    pages = load_pdf(uploaded_file)

                    if not pages:
                        st.warning(
                            f"No readable text found in {uploaded_file.name}"
                        )
                        continue

                    # Split text into chunks
                    chunks = chunk_documents(pages)

                    if not chunks:
                        st.warning(
                            f"No text chunks created for {uploaded_file.name}"
                        )
                        continue

                    # Generate embeddings
                    embeddings = create_embeddings(
                        [chunk["text"] for chunk in chunks]
                    )

                    # Save in ChromaDB
                    document_id = save_to_chroma(
                        chunks,
                        embeddings,
                        document_name=uploaded_file.name
                    )

                    document_ids.append(document_id)

                    st.success(
                        f"{uploaded_file.name} processed successfully!"
                    )

                    st.write("Pages:", len(pages))
                    st.write("Chunks:", len(chunks))

                if document_ids:

                    st.session_state.uploaded_document_ids = document_ids

                    st.success(
                        f"Successfully processed {len(document_ids)} document(s)!"
                    )

                    st.rerun()

            except Exception as error:
                st.error(f"PDF processing failed: {error}")


# --------------------------------------------------
# DOCUMENT SELECTION
# --------------------------------------------------

st.divider()

st.header("📑 Select a document")

records = uploaded_collection.get(
    include=["metadatas"]
)

available_documents = {}

for metadata in records["metadatas"]:

    document_name = metadata.get("document_name")
    document_id = metadata.get("document_id")

    if document_name and document_id:

        if document_name not in available_documents:
            available_documents[document_name] = set()

        available_documents[document_name].add(document_id)


if available_documents:

    document_options = [
        "All Documents"
    ] + sorted(available_documents.keys())

    selected_document = st.selectbox(
        "Choose which PDF to search:",
        document_options
    )

    if selected_document == "All Documents":

        selected_document_ids = [
            document_id
            for ids in available_documents.values()
            for document_id in ids
        ]

    else:

        selected_document_ids = list(
            available_documents[selected_document]
        )

    st.success(f"Selected: {selected_document}")

else:

    selected_document_ids = []
    st.info("No uploaded documents. Using original BTU document.")


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

st.divider()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View source documents"):

                for source in message["sources"]:

                    st.write(f"PDF: {source['document_name']}")
                    st.write(f"PDF Page: {source['page']}")
                    st.write(source["text"])
                    st.divider()


# --------------------------------------------------
# CHAT INPUT WITH CONVERSATION MEMORY
# --------------------------------------------------

st.header("💬 Ask your document")

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                answer, sources = generate_answer(
                    question,
                    document_ids=(
                        selected_document_ids
                        if selected_document_ids
                        else None
                    ),
                    chat_history=st.session_state.messages[:-1]
                )

            st.markdown(answer)

            if sources:

                with st.expander("View source documents"):

                    for source in sources:

                        st.write(f"PDF: {source['document_name']}")
                        st.write(f"Page: {source['page']}")
                        st.write(source["text"])
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:

            st.error(f"Something went wrong: {error}")
