import unicodedata
import re
from pathlib import Path
from collections import Counter
import sys

# -----------------------------
# Helpers
# -----------------------------

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # fallback attempt
        return path.read_text(errors="replace")

def normalize(s: str) -> str:
    """Return NFC normalized version."""
    return unicodedata.normalize("NFC", s)

def is_latin_char(ch: str) -> bool:
    """Check whether char is in Latin Unicode blocks."""
    return (
        "LATIN" in unicodedata.name(ch, "") or
        ch in " \n\r\t.,;:!?-—()[]{}'\"«»"
    )

def detect_non_latin_chars(text: str):
    """List unusual characters that might indicate OCR noise."""
    bad = Counter(ch for ch in text if not is_latin_char(ch))
    return bad

def top_tokens(text: str, n=40):
    words = re.findall(r"[A-Za-zÀ-ſ]+", text)
    freq = Counter(words).most_common(n)
    return freq

def inspect_file(path: Path):
    text = read_text(path)
    norm = normalize(text)

    print("="*80)
    print(f"FILE: {path.name}")
    print("="*80)

    # ---------------------------------------------------------
    # 1. Size overview
    # ---------------------------------------------------------
    print(f"Characters: {len(text):,}")
    print(f"Lines: {text.count(chr(10)):,}")
    print()

    # ---------------------------------------------------------
    # 2. Uppercase / lowercase distribution
    # ---------------------------------------------------------
    words = re.findall(r"[A-Za-zÀ-ſ]+", text)
    upper_words = [w for w in words if w[0].isupper()]
    lower_words = [w for w in words if w[0].islower()]

    print("WORD CASE DISTRIBUTION:")
    print(f" Total words found: {len(words):,}")
    print(f"   Words starting with CAPITAL: {len(upper_words):,}  ({len(upper_words)/len(words)*100:.2f}%)")
    print(f"   Words starting with lowercase: {len(lower_words):,}  ({len(lower_words)/len(words)*100:.2f}%)")
    print()

    # ---------------------------------------------------------
    # 3. Digits (Arabic numerals)
    # ---------------------------------------------------------
    digits = re.findall(r"\d+", text)
    print("DIGITS / ARABIC NUMERALS:")
    print(f" Total numeric sequences: {len(digits):,}")
    print(f" Examples: {digits[:10]}")
    print()

    # ---------------------------------------------------------
    # 4. Non-Latin characters
    # ---------------------------------------------------------
    bad = detect_non_latin_chars(text)
    print("NON-LATIN OR UNUSUAL CHARACTERS:")
    if bad:
        for ch, cnt in bad.most_common(20):
            code = hex(ord(ch))
            name = unicodedata.name(ch, "UNKNOWN")
            print(f"  {repr(ch)}  {cnt:,} times   (U+{code}, {name})")
    else:
        print("  ✔ No unusual characters detected.")
    print()

    # ---------------------------------------------------------
    # 5. Line length statistics
    # ---------------------------------------------------------
    lengths = [len(line) for line in text.splitlines()]
    if lengths:
        print("LINE LENGTHS:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths)/len(lengths):.2f}")
        print()

    # ---------------------------------------------------------
    # 6. Frequent tokens
    # ---------------------------------------------------------
    print("TOP FREQUENT TOKENS:")
    for w, c in top_tokens(text, 40):
        print(f"  {w:<20} {c}")
    print()

    # ---------------------------------------------------------
    # 7. Repeated header/footer pattern detection
    # ---------------------------------------------------------
    print("HEADER / FOOTER PATTERN CHECK:")
    candidates = [
        r"the latin library",
        r"classics page",
        r"copyright",
        r"www",
        r"http",
    ]
    lc = text.lower()

    for p in candidates:
        count = len(re.findall(p, lc))
        if count:
            print(f"  '{p}' appears {count} times → likely footer/header noise.")
    print()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_corpus_quality.py <file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        inspect_file(target)

    elif target.is_dir():
        for file in sorted(target.glob("*.txt")):
            inspect_file(file)

    else:
        print("Not a valid file or folder.")
