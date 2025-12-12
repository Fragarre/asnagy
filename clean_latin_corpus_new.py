import re
import unicodedata
from pathlib import Path
import sys

# ============================================================
# CLEANING STEPS
# ============================================================
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")

def normalize_unicode(text: str) -> str:
    """Normalize to NFC + expand ligatures."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("æ", "ae").replace("Æ", "Ae")
    text = text.replace("œ", "oe").replace("Œ", "Oe")
    return text

def remove_headers_footers(text: str) -> str:
    """Removes boilerplate: Latin Library, URLs, copyright."""
    patterns = [
        r"the latin library",
        r"classics page",
        r"copyright.*?\n",
        r"https?://\S+",
        r"www\.\S+",
        r"all rights reserved.*?\n",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text

def remove_arabic_numerals(text: str) -> str:
    """Remove digits (12345) but keep Roman numerals."""
    return re.sub(r"\d+", "", text)

def clean_punctuation(text: str) -> str:
    """
    Keep punctuation; remove exotic unicode marks.
    Do *not* strip normal Latin punctuation.
    """
    # convert fancy punctuation → ASCII equivalents
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    # remove weird symbols (but keep . , ; : ? ! etc.)
    text = re.sub(r"[•■◆▲►▮…]", " ", text)
    # collapse repeated punctuation (?? → ?)
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)
    return text

def remove_garbage_chars(text: str) -> str:
    """
    Allow:
      - alphabetic chars (including latin extended)
      - whitespace
      - punctuation: . , ; : ? ! ' " ( ) [ ] - —
    Replace everything else with a space.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  " .,:;!?()[]'\"-\n\r\t")
    cleaned = []
    for ch in text:
        if ch in allowed or ch.isalpha() or ch.isspace():
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return "".join(cleaned)

def remove_specific_noise(text: str) -> str:
    """Custom removable characters."""
    return (
        text.replace("µ", "")
            .replace("º", "")
            .replace("ª", "")
    )

def remove_extra_spaces(text: str) -> str:
    """Normalize spacing and newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # keep paragraph breaks
    return text.strip()

def remove_empty_lines(text: str) -> str:
    """Remove empty / void lines and collapse multiple blank lines."""
    lines = text.splitlines()
    # keep only lines containing at least one non-space character
    cleaned_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(cleaned_lines)

def remove_empty_brackets(text: str) -> str:
    """
    Remove empty () or [] even if they contain whitespace.
    Keep brackets if they contain any characters.
    """
    # Remove empty parentheses: (), ( ), (   )
    text = re.sub(r"\(\s*\)", "", text)

    # Remove empty brackets: [], [ ], [   ]
    text = re.sub(r"\[\s*\]", "", text)

    return text

def remove_no_break_spaces(text: str) -> str:
    """Remove NO-BREAK SPACE (U+00A0) and similar HTML whitespace."""
    return text.replace("\u00A0", " ")

# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_no_break_spaces(text) 
    text = remove_headers_footers(text)
    text = remove_arabic_numerals(text)
    text = clean_punctuation(text)
    text = remove_empty_brackets(text)   
    text = remove_specific_noise(text)
    text = remove_garbage_chars(text)
    text = remove_empty_lines(text)  
    text = remove_extra_spaces(text)
    return text



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_latin_corpus.py <file.txt>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print("File not found.")
        sys.exit(1)

    original = read_text(input_path)
    cleaned = clean_text(original)

    output_path = input_path.parent / f"cleaned_{input_path.name}"
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"✔ Cleaned file written to: {output_path}")
