import re
import unicodedata
from pathlib import Path
import sys

# ============================================================
# CLEANING STEPS  (UNCHANGED)
# ============================================================
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("æ", "ae").replace("Æ", "Ae")
    text = text.replace("œ", "oe").replace("Œ", "Oe")
    return text

def remove_headers_footers(text: str) -> str:
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
    return re.sub(r"\d+", "", text)

def clean_punctuation(text: str) -> str:
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[•■◆▲►▮…]", " ", text)
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)
    return text

def remove_garbage_chars(text: str) -> str:
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
    return (
        text.replace("µ", "")
            .replace("º", "")
            .replace("ª", "")
    )

def remove_extra_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def remove_empty_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(cleaned_lines)

def remove_empty_brackets(text: str) -> str:
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    return text

def remove_no_break_spaces(text: str) -> str:
    return text.replace("\u00A0", " ")

# ============================================================
# MAIN CLEANING PIPELINE (UNCHANGED)
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
# ENTRY POINT — UPDATED FOR FOLDER INPUT
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_latin_corpus.py <input_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])

    if not input_folder.exists() or not input_folder.is_dir():
        print("Error: input must be a folder.")
        sys.exit(1)

    output_folder = input_folder / "cleaned"
    output_folder.mkdir(exist_ok=True)

    txt_files = list(input_folder.glob("*.txt"))

    if not txt_files:
        print("No .txt files found in folder.")
        sys.exit(0)

    for f in txt_files:
        print(f"Cleaning: {f.name}")
        original = read_text(f)
        cleaned = clean_text(original)
        (output_folder / f.name).write_text(cleaned, encoding="utf-8")

    print(f"\n✔ All files cleaned and saved in: {output_folder}")
