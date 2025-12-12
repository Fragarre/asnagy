from pathlib import Path

def concatenate_cleaned(input_folder: str, output_file: str = "latin_all_raw.txt"):
    input_path = Path(input_folder)
    if not input_path.is_dir():
        raise ValueError(f"Not a folder: {input_folder}")

    output_path = input_path.parent / output_file

    with output_path.open("w", encoding="utf-8") as out:
        for file in sorted(input_path.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            out.write(text.strip() + "\n")

    print(f"✔ Concatenated corpus written to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python concat_cleaned.py <cleaned_folder>")
        sys.exit(1)

    concatenate_cleaned(sys.argv[1])
