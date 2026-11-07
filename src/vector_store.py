
import chromadb
import uuid


def save_to_chroma(
    chunks,
    embeddings,
    collection_name="uploaded_documents",
    document_name="Unknown PDF"
):
    """Save document chunks and replace older copies of the same PDF."""

    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name=collection_name
    )

    # Remove previous copies of this document
    existing = collection.get(
        where={"document_name": document_name}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"Removed previous copy of: {document_name}")

    # Create a unique ID for this upload
    document_id = str(uuid.uuid4())

    collection.add(
        ids=[
            f"{document_id}_chunk_{i}"
            for i in range(len(chunks))
        ],
        documents=[
            chunk["text"] for chunk in chunks
        ],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "page": chunk["page"],
                "document_id": document_id,
                "document_name": document_name,
                "section": chunk.get("section", "Unknown")
            }
            for chunk in chunks
        ]
    )

    print("Successfully saved document to ChromaDB!")
    print("Total stored chunks:", collection.count())

    return document_id
