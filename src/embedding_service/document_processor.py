import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from pypdf import PdfReader
import gc


def get_pdf_page_count(pdf_path):
    """Returns the total number of pages in the PDF file."""
    try:
        info = pdfinfo_from_path(pdf_path)
        return int(info.get("Pages", 0))
    except Exception:
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as e:
            print(f"Error reading PDF page count: {e}")
            return 0


def iter_pdf_pages(pdf_path, dpi=130, max_dim=1400, extract_text=True):
    """
    Generator that yields (page_num, total_pages, PIL Image, page_text) one page at a time.
    Keeps memory footprint constant (~25MB) regardless of PDF length.
    Extracts text per page to store rich filterable metadata alongside embeddings.
    """
    total_pages = get_pdf_page_count(pdf_path)
    print(f"Streaming PDF pages from: {pdf_path} (Total pages: {total_pages})")

    reader = None
    if extract_text:
        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"Warning: Could not initialize PdfReader for text extraction: {e}")

    for page_num in range(1, total_pages + 1):
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
            if not pages:
                continue
            img = pages[0]
            w, h = img.size
            if w > max_dim or h > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            page_text = ""
            if reader and page_num <= len(reader.pages):
                try:
                    page_text = reader.pages[page_num - 1].extract_text() or ""
                except Exception:
                    pass

            yield page_num, total_pages, img, page_text.strip()
            del pages
            del img
            gc.collect()
        except Exception as e:
            print(f"Error converting page {page_num}: {e}")
            continue


def pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of PIL Images, one per page (buffered).
    """
    return [img for _, _, img, _ in iter_pdf_pages(pdf_path)]


def image_to_base64(image):
    """
    Converts a PIL Image to a Base64 encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_pdf_page_to_base64(pdf_path, page_num, dpi=130, max_dim=1400):
    """
    Renders a specific page of a PDF file to a Base64-encoded JPEG image.
    Useful for resolving image context on demand from metadata during generation.
    """
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
        if not pages:
            return None
        img = pages[0]
        w, h = img.size
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image_to_base64(img)
    except Exception as e:
        print(f"Error rendering page {page_num} of {pdf_path}: {e}")
        return None
