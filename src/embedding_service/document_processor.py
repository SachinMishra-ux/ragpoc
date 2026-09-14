import os
import sys
import gc
import re
import shutil
import base64
import tempfile
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Generator, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont

# Optional PDF libraries
try:
    from pdf2image import convert_from_path, pdfinfo_from_path
except ImportError:
    convert_from_path = None
    pdfinfo_from_path = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}


# ---------------------------------------------------------------------------
# 1. Base64 & Image Utilities
# ---------------------------------------------------------------------------
def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded JPEG string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _get_default_font(size: int = 24):
    """Attempts to load a standard TrueType font on macOS / Linux, falls back to default."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars_per_line: int = 75) -> List[str]:
    """Wraps paragraphs cleanly into lines of at most max_chars_per_line."""
    lines = []
    for paragraph in text.split("\n"):
        p = paragraph.strip()
        if not p:
            lines.append("")
            continue
        words = p.split(" ")
        current_line = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 <= max_chars_per_line:
                current_line.append(w)
                current_len += len(w) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [w]
                current_len = len(w)
        if current_line:
            lines.append(" ".join(current_line))
    return lines


def render_document_card_image(
    doc_name: str,
    page_num: int,
    total_pages: int,
    text_content: str,
    title: Optional[str] = None,
    doc_type: str = "WORD DOCUMENT",
    embedded_img: Optional[Image.Image] = None,
    width: int = 1200,
    height: int = 1500,
) -> Image.Image:
    """
    Renders a high-resolution, readable academic visual page card for text/word documents.
    Enables multimodal embedding models (Gemini Embedding 2) to extract visual layout and text.
    """
    img = Image.new("RGB", (width, height), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_header = _get_default_font(28)
    font_title = _get_default_font(34)
    font_body = _get_default_font(22)
    font_meta = _get_default_font(20)

    # Header Bar
    draw.rectangle([(0, 0), (width, 85)], fill=(30, 41, 59))  # Dark Slate
    draw.text((40, 28), f"PAGE: {doc_type.upper()} - {doc_name}", fill=(241, 245, 249), font=font_header)
    page_badge = f"Page {page_num} of {total_pages}"
    draw.text((width - 240, 30), page_badge, fill=(148, 163, 184), font=font_meta)

    # Card Border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(226, 232, 240), width=2)

    current_y = 120

    # Optional Title
    if title:
        draw.text((50, current_y), title, fill=(15, 23, 42), font=font_title)
        current_y += 55
        draw.line([(50, current_y), (width - 50, current_y)], fill=(203, 213, 225), width=2)
        current_y += 30

    # Optional Embedded Image from document
    if embedded_img:
        try:
            emb_w, emb_h = embedded_img.size
            max_img_w = width - 100
            max_img_h = 400
            scale = min(max_img_w / emb_w, max_img_h / emb_h, 1.0)
            target_w = int(emb_w * scale)
            target_h = int(emb_h * scale)
            resized_emb = embedded_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img.paste(resized_emb, (50, current_y))
            current_y += target_h + 30
        except Exception as e:
            print(f"Notice: Failed to render embedded image: {e}")

    # Body Text
    wrapped_lines = _wrap_text(text_content, max_chars_per_line=80)
    for line in wrapped_lines:
        if current_y > height - 80:
            draw.text((50, current_y), "... [Content continues on next page] ...", fill=(100, 116, 139), font=font_meta)
            break
        if line == "":
            current_y += 16
        else:
            draw.text((50, current_y), line, fill=(30, 41, 59), font=font_body)
            current_y += 30

    # Footer
    draw.line([(50, height - 60), (width - 50, height - 60)], fill=(226, 232, 240), width=1)
    draw.text((50, height - 48), f"Indexed in Amazon S3 Vectors • {doc_name}", fill=(148, 163, 184), font=font_meta)

    return img


def render_slide_card_image(
    doc_name: str,
    slide_num: int,
    total_slides: int,
    slide_title: str,
    bullet_points: List[str],
    embedded_img: Optional[Image.Image] = None,
    width: int = 1400,
    height: int = 900,
) -> Image.Image:
    """
    Renders a high-resolution academic presentation slide card (16:9 / 4:3 aspect ratio).
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Fonts
    font_header = _get_default_font(26)
    font_title = _get_default_font(38)
    font_bullet = _get_default_font(24)
    font_meta = _get_default_font(20)

    # Top Presentation Banner
    draw.rectangle([(0, 0), (width, 80)], fill=(37, 99, 235))  # Royal Blue
    draw.text((40, 24), f"PRESENTATION: {doc_name}", fill=(255, 255, 255), font=font_header)
    slide_badge = f"Slide {slide_num} of {total_slides}"
    draw.text((width - 240, 26), slide_badge, fill=(219, 234, 254), font=font_meta)

    # Outer border
    draw.rectangle([(15, 15), (width - 15, height - 15)], outline=(203, 213, 225), width=2)

    current_y = 120

    # Slide Title
    title_text = slide_title if slide_title else f"Slide {slide_num}"
    draw.text((60, current_y), title_text, fill=(15, 23, 42), font=font_title)
    current_y += 60
    draw.line([(60, current_y), (width - 60, current_y)], fill=(59, 130, 246), width=3)
    current_y += 35

    # If slide has an embedded image, place it on the right side
    content_width = width - 120
    if embedded_img:
        try:
            emb_w, emb_h = embedded_img.size
            max_img_w = 480
            max_img_h = height - current_y - 100
            scale = min(max_img_w / emb_w, max_img_h / emb_h, 1.0)
            target_w = int(emb_w * scale)
            target_h = int(emb_h * scale)
            resized_emb = embedded_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_x = width - target_w - 60
            img.paste(resized_emb, (img_x, current_y))
            content_width = width - target_w - 150
        except Exception as e:
            print(f"Notice: Failed to render slide image: {e}")

    # Bullet Points
    max_chars = int(content_width / 13)
    for bp in bullet_points:
        if current_y > height - 100:
            break
        bp_clean = bp.strip()
        if not bp_clean:
            continue
        wrapped = _wrap_text(bp_clean, max_chars_per_line=max_chars)
        # Bullet circle
        draw.ellipse([(60, current_y + 8), (72, current_y + 20)], fill=(37, 99, 235))
        for line_idx, line in enumerate(wrapped):
            if current_y > height - 80:
                break
            indent = 85
            draw.text((indent, current_y), line, fill=(30, 41, 59), font=font_bullet)
            current_y += 34
        current_y += 12

    # Footer
    draw.line([(60, height - 50), (width - 60, height - 50)], fill=(226, 232, 240), width=1)
    draw.text((60, height - 38), "Academic Presentation • S3 Vectors Index", fill=(148, 163, 184), font=font_meta)

    return img


# ---------------------------------------------------------------------------
# 2. LibreOffice Headless PDF Conversion (Tier 1)
# ---------------------------------------------------------------------------
def _find_libreoffice_binary() -> Optional[str]:
    """Finds LibreOffice/soffice executable on macOS or Linux."""
    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/usr/bin/soffice",
        "/usr/bin/libreoffice",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def convert_to_pdf_via_libreoffice(file_path: str, output_dir: str) -> Optional[str]:
    """
    Attempts to convert a Word/PowerPoint document to PDF using headless LibreOffice.
    Returns path to converted PDF, or None if LibreOffice is unavailable or fails.
    """
    soffice = _find_libreoffice_binary()
    if not soffice:
        return None

    try:
        cmd = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            file_path,
            "--outdir",
            output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            base = os.path.splitext(os.path.basename(file_path))[0]
            expected_pdf = os.path.join(output_dir, f"{base}.pdf")
            if os.path.exists(expected_pdf):
                return expected_pdf
    except Exception as e:
        print(f"LibreOffice conversion error: {e}")

    return None


# ---------------------------------------------------------------------------
# 3. Native Parsing for Word Documents (.docx, .doc) (Tier 2)
# ---------------------------------------------------------------------------
def _extract_docx_paragraphs(docx_path: str) -> List[str]:
    """Extracts text paragraphs from a .docx file using standard library zipfile & XML."""
    paragraphs = []
    try:
        with zipfile.ZipFile(docx_path, "r") as z:
            if "word/document.xml" in z.namelist():
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                for p_node in tree.iterfind(".//w:p", ns):
                    texts = [t_node.text for t_node in p_node.iterfind(".//w:t", ns) if t_node.text]
                    if texts:
                        paragraphs.append("".join(texts))
    except Exception as e:
        print(f"Error parsing .docx with standard XML: {e}")

    # Fallback to macOS textutil if empty
    if not paragraphs and sys.platform == "darwin":
        try:
            res = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", docx_path], capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout:
                paragraphs = [p for p in res.stdout.split("\n\n") if p.strip()]
        except Exception:
            pass

    return paragraphs


def _extract_doc_text(doc_path: str) -> List[str]:
    """Extracts text from legacy .doc files using macOS textutil or strings."""
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", doc_path], capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout:
                return [p for p in res.stdout.split("\n\n") if p.strip()]
        except Exception as e:
            print(f"textutil failed for {doc_path}: {e}")

    try:
        with open(doc_path, "rb") as f:
            raw = f.read()
        extracted = "".join(chr(b) if 32 <= b <= 126 or b in (10, 13) else " " for b in raw)
        lines = [line.strip() for line in extracted.split("\n") if len(line.strip()) > 3]
        return lines
    except Exception:
        return []


def iter_word_pages(file_path: str) -> Generator[Tuple[int, int, Image.Image, str], None, None]:
    """
    Generator yielding (page_num, total_pages, PIL_Image, page_text) for .docx or .doc files.
    Paginates paragraphs logically (~350 words per page) and renders academic card images.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        paragraphs = _extract_docx_paragraphs(file_path)
        doc_type = "WORD DOCUMENT (.DOCX)"
    else:
        paragraphs = _extract_doc_text(file_path)
        doc_type = "WORD DOCUMENT (.DOC)"

    if not paragraphs:
        paragraphs = [f"Empty or unreadable document: {filename}"]

    # Group into logical pages (~350 words per page)
    pages_content: List[str] = []
    current_page: List[str] = []
    current_word_count = 0

    for p in paragraphs:
        p_words = len(p.split())
        if current_word_count + p_words > 350 and current_page:
            pages_content.append("\n\n".join(current_page))
            current_page = [p]
            current_word_count = p_words
        else:
            current_page.append(p)
            current_word_count += p_words

    if current_page:
        pages_content.append("\n\n".join(current_page))

    total_pages = len(pages_content)
    print(f"Streaming Word document {filename} (Logical pages: {total_pages})")

    for page_num, text_chunk in enumerate(pages_content, start=1):
        img = render_document_card_image(
            doc_name=filename,
            page_num=page_num,
            total_pages=total_pages,
            text_content=text_chunk,
            title=f"Section {page_num}" if total_pages > 1 else None,
            doc_type=doc_type,
        )
        yield page_num, total_pages, img, text_chunk


# ---------------------------------------------------------------------------
# 4. Native Parsing for PowerPoint Presentations (.pptx, .ppt) (Tier 2)
# ---------------------------------------------------------------------------
def _extract_pptx_slides(pptx_path: str) -> List[Tuple[str, List[str]]]:
    """
    Parses a .pptx file using standard zipfile & XML to extract slides in sequential order.
    Returns a list of (slide_title, bullet_points).
    """
    slides_data = []
    try:
        with zipfile.ZipFile(pptx_path, "r") as z:
            slide_names = [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n)]
            # Sort slides numerically (slide1.xml, slide2.xml, ..., slide10.xml)
            slide_names.sort(key=lambda x: int(re.search(r"slide(\d+)\.xml", x).group(1)))

            ns = {
                "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            }

            for s_name in slide_names:
                xml_data = z.read(s_name)
                tree = ET.fromstring(xml_data)

                # Extract text blocks
                paragraphs = []
                for p_node in tree.iterfind(".//a:p", ns):
                    p_text = "".join([t_node.text for t_node in p_node.iterfind(".//a:t", ns) if t_node.text]).strip()
                    if p_text:
                        paragraphs.append(p_text)

                if paragraphs:
                    title = paragraphs[0]
                    bullets = paragraphs[1:] if len(paragraphs) > 1 else []
                else:
                    title = f"Slide {len(slides_data) + 1}"
                    bullets = []

                slides_data.append((title, bullets))
    except Exception as e:
        print(f"Error parsing .pptx slides: {e}")

    return slides_data


def iter_powerpoint_slides(file_path: str) -> Generator[Tuple[int, int, Image.Image, str], None, None]:
    """
    Generator yielding (slide_num, total_slides, PIL_Image, slide_text) for .pptx or .ppt files.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pptx":
        slides = _extract_pptx_slides(file_path)
    else:
        text_lines = _extract_doc_text(file_path)
        slides = []
        for i in range(0, max(len(text_lines), 1), 5):
            chunk = text_lines[i : i + 5]
            title = chunk[0] if chunk else f"Slide {len(slides) + 1}"
            bullets = chunk[1:] if len(chunk) > 1 else []
            slides.append((title, bullets))

    if not slides:
        slides = [(f"Slide Presentation: {filename}", [f"No text extracted from {filename}"])]

    total_slides = len(slides)
    print(f"Streaming PowerPoint {filename} (Total slides: {total_slides})")

    for slide_num, (title, bullets) in enumerate(slides, start=1):
        combined_text = f"Title: {title}\n" + "\n".join([f"- {b}" for b in bullets])
        img = render_slide_card_image(
            doc_name=filename,
            slide_num=slide_num,
            total_slides=total_slides,
            slide_title=title,
            bullet_points=bullets,
        )
        yield slide_num, total_slides, img, combined_text.strip()


# ---------------------------------------------------------------------------
# 5. Standard PDF Processor
# ---------------------------------------------------------------------------
def get_pdf_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in the PDF file."""
    if pdfinfo_from_path:
        try:
            info = pdfinfo_from_path(pdf_path)
            return int(info.get("Pages", 0))
        except Exception:
            pass
    if PdfReader:
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as e:
            print(f"Error reading PDF page count: {e}")
    return 0


def iter_pdf_pages(pdf_path: str, dpi: int = 130, max_dim: int = 1400, extract_text: bool = True):
    """
    Generator that yields (page_num, total_pages, PIL Image, page_text) one page at a time.
    Keeps memory footprint constant (~25MB) regardless of PDF length.
    """
    total_pages = get_pdf_page_count(pdf_path)
    print(f"Streaming PDF pages from: {pdf_path} (Total pages: {total_pages})")

    reader = None
    if extract_text and PdfReader:
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


def pdf_to_images(pdf_path: str):
    """Converts a PDF file into a list of PIL Images, one per page (buffered)."""
    return [img for _, _, img, _ in iter_pdf_pages(pdf_path)]


def render_pdf_page_to_base64(pdf_path: str, page_num: int, dpi: int = 130, max_dim: int = 1400):
    """Renders a specific page of a PDF file to a Base64-encoded JPEG image."""
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


# ---------------------------------------------------------------------------
# 6. Universal Multi-Format Document Streamer
# ---------------------------------------------------------------------------
def get_document_page_count(file_path: str) -> int:
    """Returns the page / slide count for any supported document format."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return get_pdf_page_count(file_path)
    elif ext in {".docx", ".doc"}:
        return sum(1 for _ in iter_word_pages(file_path))
    elif ext in {".pptx", ".ppt"}:
        return sum(1 for _ in iter_powerpoint_slides(file_path))
    return 0


def iter_document_pages(
    file_path: str,
    dpi: int = 130,
    max_dim: int = 1400,
    extract_text: bool = True,
) -> Generator[Tuple[int, int, Image.Image, str], None, None]:
    """
    Universal multi-format generator that yields (page_num, total_pages, PIL Image, page_text)
    for PDF, Word (.docx, .doc), and PowerPoint (.pptx, .ppt) documents.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        yield from iter_pdf_pages(file_path, dpi=dpi, max_dim=max_dim, extract_text=extract_text)
        return

    # For Office files (.docx, .doc, .pptx, .ppt):
    # Attempt Tier 1: Headless LibreOffice conversion to PDF
    tmp_dir = tempfile.mkdtemp(prefix="office_conv_")
    try:
        converted_pdf = convert_to_pdf_via_libreoffice(file_path, tmp_dir)
        if converted_pdf and os.path.exists(converted_pdf):
            print(f"Converted {file_path} to PDF via LibreOffice: {converted_pdf}")
            yield from iter_pdf_pages(converted_pdf, dpi=dpi, max_dim=max_dim, extract_text=extract_text)
            return
    except Exception as e:
        print(f"LibreOffice conversion fallback triggered: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Tier 2: Built-in Native Parsers + PIL Card Rendering
    if ext in {".docx", ".doc"}:
        yield from iter_word_pages(file_path)
    elif ext in {".pptx", ".ppt"}:
        yield from iter_powerpoint_slides(file_path)
    else:
        raise ValueError(f"Unsupported document format: '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}")
