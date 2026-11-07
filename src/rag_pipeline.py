
import chromadb
from src.embeddings import model
from langchain_ollama import ChatOllama


# --------------------------------------------------
# LOCAL LANGUAGE MODEL
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)


# --------------------------------------------------
# RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def retrieve_relevant_chunks(
    question,
    n_results=8,
    document_ids=None
):
    """Retrieve relevant passages from uploaded PDFs."""

    client = chromadb.PersistentClient(path="chroma_db")

    question_embedding = model.encode([question]).tolist()

    # Search uploaded PDFs
    if document_ids:

        collection = client.get_collection(
            name="uploaded_documents"
        )

        results_filter = {
            "document_id": {
                "$in": document_ids
            }
        }

    # Search original BTU document
    else:

        collection = client.get_collection(
            name="neural_signal_documents"
        )

        results_filter = None

    if collection.count() == 0:
        return []

    query_options = {
        "query_embeddings": question_embedding,
        "n_results": min(n_results, collection.count()),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if results_filter:
        query_options["where"] = results_filter

    results = collection.query(**query_options)

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][i]

        retrieved_chunks.append({
            "text": results["documents"][0][i],
            "page": metadata["page"],
            "document_id": metadata.get("document_id"),
            "document_name": metadata.get(
                "document_name",
                "BTU Neural Signal Document"
            ),
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# --------------------------------------------------
# GENERATE ANSWER WITH CONVERSATION MEMORY
# --------------------------------------------------

def generate_answer(
    question,
    document_ids=None,
    chat_history=None
):
    """Generate grounded answers using PDF context and chat history."""

    chunks = retrieve_relevant_chunks(
        question,
        document_ids=document_ids
    )

    if not chunks:

        return (
            "I could not find relevant information in the selected documents.",
            []
        )

    # Prepare document context
    context = ""

    for chunk in chunks:

        context += (
            f"\n[PDF: {chunk['document_name']}, "
            f"Page: {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    # Prepare conversation history
    history_text = ""

    if chat_history:

        for message in chat_history[-6:]:

            role = message["role"].upper()
            content = message["content"]

            history_text += (
                f"\n{role}: {content}\n"
            )

    # Build prompt
    prompt = f"""
You are an enterprise AI knowledge assistant.

Use the PDF context and conversation history to answer
the user's question.

RULES:
1. Use only the provided PDF context for factual answers.
2. Use conversation history to understand follow-up questions.
3. Explain answers in simple, clear language.
4. Do not invent information.
5. If the answer is unavailable in the PDF context, say:
   "I could not find this information in the selected documents."
6. Mention the PDF filename and page number used.
7. Conversation history is for understanding context, not a
   replacement for evidence from the PDFs.

CONVERSATION HISTORY:
{history_text}

PDF CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

ANSWER:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content, chunks
