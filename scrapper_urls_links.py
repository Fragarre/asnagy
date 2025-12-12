import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import unicodedata
import time

# -----------------------------
# Funciones de limpieza
# -----------------------------
def remove_footer_noise(text: str) -> str:
    """Elimina textos repetitivos / footers típicos de The Latin Library."""
    patterns = [
        r"the latin library",
        r"the classics page",
        r"copyright.*",
        r"all rights reserved.*",
        r"www\.thelatinlibrary\.com",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    # limpiar saltos de línea extra
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def remove_greek_chars(text: str) -> str:
    """
    Elimina caracteres griegos y sus diacríticos combinantes inmediatos.
    Bloques cubiertos:
      - Greek and Coptic: U+0370-U+03FF
      - Greek Extended:   U+1F00-U+1FFF
    Además elimina diacríticos combinantes que sigan a esos caracteres
    (rango U+0300-U+036F).
    """
    # Primero normalizamos a NFD para separar base + combinantes
    text_nfd = unicodedata.normalize("NFD", text)
    # Eliminamos base griega + posibles combinantes inmediatamente siguientes
    text_nofreek = re.sub(r'[\u0370-\u03FF\u1F00-\u1FFF][\u0300-\u036F]*', ' ', text_nfd)
    # Volvemos a normalizar a NFC
    return unicodedata.normalize("NFC", text_nofreek)

def basic_utf_noise_fix(text: str) -> str:
    replacements = [
        'â\x80\x9c', 'â\x80\x9d',
        'â\x80\x94', 'â\x80\x99',
        'â\x80\x98', '\r\n',
        '_'
    ]
    for s in replacements:
        text = text.replace(s, " ")

    # IMPORTANT: do NOT remove [ ]  → keep punctuation
    # text = text.replace("[", "").replace("]", "")

    text = re.sub(r'[ \t]+', ' ', text)
    return text

def remove_beta_like_noise(text: str) -> str:
    """Remove Beta Code noise without touching Latin punctuation."""
    # remove only patterns like a) e)
    text = re.sub(r'\b[A-Za-z]\)\b', ' ', text)

    # keep parentheses; remove only slash artifacts
    text = re.sub(r'[\/]{2,}', ' ', text)
    text = re.sub(r'(?<=\w)[\/]+(?=\w)', ' ', text)

    return text


def remove_arabic_numerals(text: str) -> str:
    """Remove digits only."""
    return re.sub(r'\d+', ' ', text)


def final_cleanup(text: str) -> str:
    """Llamada única para limpiar el texto según tus requisitos."""
    # 1) arreglos básicos de UTF y ruido
    text = basic_utf_noise_fix(text)

    # 2) eliminar footers y pie de página comunes
    text = remove_footer_noise(text)

    # 3) eliminar caracteres griegos (y diacríticos combinantes)
    text = remove_greek_chars(text)

    # 4) eliminar restos de Beta Code-like que rompen palabras (opcional, pero útil)
    text = remove_beta_like_noise(text)

    # 5) eliminar números arábigos (mantener numerales romanos)
    text = remove_arabic_numerals(text)

    # 6) normalización final: colapsar espacios y mantener saltos de párrafo
    # preservamos dobles saltos como separación de párrafos
    text = re.sub(r'[ \u00A0]{2,}', ' ', text)                  # espacios repetidos
    text = re.sub(r'\n\s*\n+', '\n\n', text)                    # múltiples párrafos -> 2 saltos
    # limpiar espacios extra en líneas
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line.strip() != "")
    # colapsar espacios interiores
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

# -----------------------------
# Utilidades de scraping
# -----------------------------
# def url_to_filename(url: str, out_dir: str = ".") -> str:
#     parsed = urlparse(url)
#     path_parts = parsed.path.strip("/").split("/")
#     if len(path_parts) >= 2:
#         author = path_parts[-2]
#     else:
#         author = "unknown"
#     base = os.path.basename(parsed.path) or "index"
#     text_name = os.path.splitext(base)[0]
#     author = re.sub(r'[^0-9A-Za-z_\-]', '_', author)
#     text_name = re.sub(r'[^0-9A-Za-z_\-]', '_', text_name)
#     filename = f"{author}_{text_name}.txt"
#     return os.path.join(out_dir, filename)

def url_to_filename(url: str, out_dir: str = ".") -> str:
    parsed = urlparse(url)

    # extract author from the path
    path_parts = parsed.path.strip("/").split("/")
    author = path_parts[-2] if len(path_parts) >= 2 else "unknown"

    # base name without extension
    base = os.path.basename(parsed.path) or "index"
    text_name = os.path.splitext(base)[0]

    # normalize author and text_name
    author = re.sub(r'[^0-9A-Za-z_\-]', '_', author)
    text_name = re.sub(r'[^0-9A-Za-z_\-]', '_', text_name)

    # handle fragment, e.g. #3 → fragment="3"
    fragment = parsed.fragment
    if fragment:
        filename = f"{author}_{text_name}_{fragment}.txt"
    else:
        filename = f"{author}_{text_name}.txt"

    return os.path.join(out_dir, filename)

def scrape_url(url: str, session: requests.Session = None, timeout: int = 15) -> str:
    """Descarga y extrae el texto principal de una URL (usa <p> como primario)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LatinLibraryScraper/1.0)"
    }
    if session is None:
        session = requests.Session()

    resp = session.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Intentar lxml si está, si no, fallback a html.parser
    try:
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        soup = BeautifulSoup(resp.text, "html.parser")

    # Preferimos los párrafos <p>, si no hay, tomamos body
    paragraphs = soup.find_all("p")
    if paragraphs:
        raw = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    else:
        raw = soup.get_text("\n\n", strip=True)

    cleaned = final_cleanup(raw)
    return cleaned

# -----------------------------
# Main: procesar fichero de enlaces
# -----------------------------
def process_links_file(links_file: str, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    with open(links_file, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            print(f"Scraping {url} ...")
            try:
                text = scrape_url(url, session=session)
                filename = url_to_filename(url, out_dir=out_dir)
                with open(filename, "w", encoding="utf-8") as out:
                    out.write(text)
                print(f"Saved → {filename}")
                # pequeña pausa para no sobrecargar el servidor
                time.sleep(0.5)
            except Exception as e:
                print(f"Error with {url}: {e}")

# Ejecución por defecto
if __name__ == "__main__":
    process_links_file("links.txt")
