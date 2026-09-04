from .pdf_loader import extract_pdf_text
from .docx_loader import extract_docx_text
from .csv_loader import extract_csv_text
from .text_loader import extract_txt_text


def extract_text(file):

    file_type = file.name.lower().split(".")[-1]

    if file_type == "pdf":
        return extract_pdf_text(file)

    elif file_type == "docx":
        return extract_docx_text(file)

    elif file_type == "csv":
        return extract_csv_text(file)

    elif file_type == "txt":
        return extract_txt_text(file)

    else:
        raise ValueError("Unsupported file type")