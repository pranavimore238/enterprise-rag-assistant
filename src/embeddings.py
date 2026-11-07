

import re
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_documents(pages, chunk_size=900, chunk_overlap=150):
    """Split PDF text into meaningful chunks while preserving section headings."""

    chunks = []

    section_pattern = re.compile(
        r"(?im)^(EDUCATION|PROJECTS|CERTIFICATIONS|"
        r"PROFESSIONAL EXPERIENCE|TECHNICAL SKILLS|"
        r"PROFESSIONAL SUMMARY|AVAILABILITY)\s*$"
    )

    for page in pages:
        text = page["text"]

        # Identify section headings
        matches = list(section_pattern.finditer(text))

        if not matches:
            # Fallback for pages without recognised headings
            start = 0

            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page["page"]
                    })

                start += chunk_size - chunk_overlap

            continue

        # Preserve each section as a separate text unit
        for index, match in enumerate(matches):
            section_start = match.start()

            if index + 1 < len(matches):
                section_end = matches[index + 1].start()
            else:
                section_end = len(text)

            section_text = text[section_start:section_end].strip()
            section_name = match.group(1)

            if not section_text:
                continue

            # Split very long sections into smaller chunks
            start = 0

            while start < len(section_text):
                end = start + chunk_size
                chunk_text = section_text[start:end]

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page["page"],
                        "section": section_name
                    })

                if end >= len(section_text):
                    break

                start += chunk_size - chunk_overlap

    return chunks


def create_embeddings(text_chunks):
    """Convert text into numerical embeddings."""

    embeddings = model.encode(text_chunks)

    return embeddings
