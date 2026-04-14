# resume_parser.py
# ─────────────────────────────────────────────────────────────────────────────
# Resume Parsing & NLP Preprocessing Module
# Handles: PDF text extraction, text cleaning, tokenisation, stopword removal,
#          named-entity extraction (name, email, phone, LinkedIn).
# ─────────────────────────────────────────────────────────────────────────────

import re
import io
import string
from typing import Optional

import pdfplumber
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK resource bootstrap ───────────────────────────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt",         "punkt"),
    ("tokenizers/punkt_tab",     "punkt_tab"),
    ("corpora/stopwords",        "stopwords"),
    ("corpora/wordnet",          "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
]

def _ensure_nltk_resources() -> None:
    for path, pkg in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_resources()

# ── Constants ─────────────────────────────────────────────────────────────────
_STOP_WORDS   = set(stopwords.words("english"))
_LEMMATIZER   = WordNetLemmatizer()

# Common resume section headings to help with section detection
SECTION_HEADINGS = [
    "education", "experience", "skills", "projects", "certifications",
    "awards", "publications", "summary", "objective", "work history",
    "professional experience", "technical skills", "contact",
]

# Regex patterns
_EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE   = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[a-zA-Z0-9\-_]+", re.IGNORECASE)
_URL_RE      = re.compile(r"https?://\S+|www\.\S+")


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_source) -> str:
    """
    Extract all text from a PDF.

    Args:
        file_source: A file path (str) OR a bytes/BytesIO object.

    Returns:
        Concatenated plain text from all pages.

    Raises:
        ValueError: If no text could be extracted.
    """
    try:
        if isinstance(file_source, (bytes, bytearray)):
            file_source = io.BytesIO(file_source)

        pages_text = []
        with pdfplumber.open(file_source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)

        full_text = "\n".join(pages_text).strip()
        if not full_text:
            raise ValueError("No readable text found in the PDF. It may be scanned/image-based.")
        return full_text

    except Exception as exc:
        raise ValueError(f"PDF extraction failed: {exc}") from exc


# ── Contact Info Extraction ───────────────────────────────────────────────────

def extract_email(text: str) -> Optional[str]:
    match = _EMAIL_RE.search(text)
    return match.group() if match else None


def extract_phone(text: str) -> Optional[str]:
    match = _PHONE_RE.search(text)
    return match.group().strip() if match else None


def extract_linkedin(text: str) -> Optional[str]:
    match = _LINKEDIN_RE.search(text)
    return match.group() if match else None


def extract_name(text: str) -> Optional[str]:
    """
    Heuristic: the candidate's name is usually on the first non-empty line
    and does NOT look like an email, URL, or heading keyword.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are clearly not names
        if _EMAIL_RE.search(line) or _URL_RE.search(line):
            continue
        if any(h in line.lower() for h in SECTION_HEADINGS):
            continue
        # A name line is typically short and title-cased
        words = line.split()
        if 1 < len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
            return line
    return None


# ── Text Cleaning & NLP Preprocessing ────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning: strip URLs, extra whitespace, and non-ASCII characters.
    Preserves casing and punctuation (needed before tokenisation).
    """
    text = _URL_RE.sub(" ", text)          # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Word-tokenise text using NLTK punkt tokeniser."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stopwords and punctuation tokens."""
    return [
        t for t in tokens
        if t.lower() not in _STOP_WORDS and t not in string.punctuation
    ]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatise tokens to their base form."""
    return [_LEMMATIZER.lemmatize(t.lower()) for t in tokens]


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      clean → tokenise → remove stopwords → lemmatise → rejoin.

    Returns a single cleaned string suitable for TF-IDF or ML features.
    """
    cleaned   = clean_text(text)
    tokens    = tokenize(cleaned)
    tokens    = remove_stopwords(tokens)
    tokens    = lemmatize(tokens)
    return " ".join(tokens)


def get_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return sent_tokenize(text)


# ── Master Parse Function ─────────────────────────────────────────────────────

def parse_resume(file_source) -> dict:
    """
    Full resume parsing pipeline.

    Args:
        file_source: PDF file path, bytes, or BytesIO.

    Returns:
        Dict with keys:
          raw_text, cleaned_text, preprocessed_text,
          name, email, phone, linkedin
    """
    raw_text        = extract_text_from_pdf(file_source)
    cleaned_text    = clean_text(raw_text)
    preprocessed    = preprocess_text(cleaned_text)

    return {
        "raw_text":          raw_text,
        "cleaned_text":      cleaned_text,
        "preprocessed_text": preprocessed,
        "name":              extract_name(raw_text),
        "email":             extract_email(raw_text),
        "phone":             extract_phone(raw_text),
        "linkedin":          extract_linkedin(raw_text),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume.pdf>")
        sys.exit(1)

    result = parse_resume(sys.argv[1])
    print("=== Parsed Resume ===")
    for k, v in result.items():
        if k in ("raw_text", "preprocessed_text", "cleaned_text"):
            print(f"{k}: {str(v)[:200]}...")
        else:
            print(f"{k}: {v}")