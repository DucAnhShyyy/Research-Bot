# tests/test_ingest.py
from src.ingest import chunk_text, extract_text_from_pdf
import fitz

def test_chunk_text_basic():
    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, chunk_size=300, overlap=100)
    assert len(chunks) > 5
    assert "text" in chunks[0]

def test_extract_text_from_pdf(tmp_path):
    # create a simple PDF
    pdf_file = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF test!")
    doc.save(pdf_file)
    doc.close()

    extracted = extract_text_from_pdf(str(pdf_file))
    assert "Hello" in extracted