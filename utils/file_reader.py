from pathlib import Path
import click
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
    Extracts text content from various file types.
    Returns the filename and its text content.
    """
    suffix = file_path.suffix.lower()
    text_content = ""

    try:
        if suffix in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        elif suffix == ".pdf":
            if PyPDF2 is None:
                click.echo("Warning: 'PyPDF2' not installed. Skipping PDF. Run 'pip install PyPDF2'.", err=True)
                return file_path.name, None
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
        elif suffix == ".docx":
            if docx is None:
                click.echo("Warning: 'python-docx' not installed. Skipping DOCX. Run 'pip install python-docx'.", err=True)
                return file_path.name, None
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text_content += para.text + "\n"
        else:
            click.echo(f"Warning: Unsupported file type '{suffix}'. Skipping '{file_path.name}'.", err=True)
            return file_path.name, None

        return file_path.name, text_content

    except Exception as e:
        click.echo(f"Error reading file {file_path.name}: {e}", err=True)
        return file_path.name, None