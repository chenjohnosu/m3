from pathlib import Path
import click
import unicodedata # Import the unicodedata module

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

def read_file(file_path: Path):
    """
    Extracts and cleans text content from various file types.
    Returns the filename and its cleaned text content.
    """
    suffix = file_path.suffix.lower()
    text_content = ""
    raw_text = ""

    try:
        if suffix == ".txt" or suffix == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        elif suffix == ".pdf":
            if PyPDF2 is None:
                click.echo("Warning: 'PyPDF2' package not found. Skipping PDF file.", err=True)
                return file_path.name, None
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                pdf_texts = [page.extract_text() for page in reader.pages]
                raw_text = "\n".join(pdf_texts)
        elif suffix == ".docx":
            if docx is None:
                click.echo("Warning: 'python-docx' package not found. Skipping DOCX file.", err=True)
                return file_path.name, None
            doc = docx.Document(file_path)
            docx_texts = [para.text for para in doc.paragraphs]
            raw_text = "\n".join(docx_texts)
        else:
            click.echo(f"Warning: Unsupported file type '{suffix}'. Skipping.", err=True)
            return file_path.name, None

        # --- AGGRESSIVE CLEANING STEP ---
        # Apply the cleaning to the raw text extracted from any file type.
        if raw_text:
            text_content = "".join(ch for ch in raw_text if unicodedata.category(ch)[0] != "C")

        return file_path.name, text_content

    except Exception as e:
        click.echo(f"Error reading file {file_path.name}: {e}", err=True)
        return file_path.name, None
