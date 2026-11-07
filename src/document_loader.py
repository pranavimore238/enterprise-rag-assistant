from pypdf import PdfReader


def load_pdf(file):
    """
    Extract text from an uploaded PDF document.
    """

    reader = PdfReader(file)

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    return extracted_pages