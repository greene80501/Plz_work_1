import os, sys, shutil, json, time, csv, hashlib, mimetypes, argparse, re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import Counter

# --- File Extraction (No changes needed here) ---
from pypdf import PdfReader
from docx import Document
from PIL import Image, ExifTags
import chardet

try:
    import openpyxl
except Exception:
    openpyxl = None
try:
    import easyocr
except Exception:
    easyocr = None

from openai import OpenAI

# --- MODIFIED: Simplified primary categories ---
PRIMARY_CATEGORIES = ["Personal", "Work"]

# --- NEW: Defined list of potential work subcategories to guide the AI ---
WORK_SUBCATEGORIES_EXAMPLES = [
    "Reports", "Invoices & Finance", "Presentations", "Meeting Notes",
    "Legal & Contracts", "Code & Scripts", "Research", "Project Plans", "Design Mockups"
]

STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the their this to was were will with without your you we our
pdf doc docx ppt pptx xls xlsx csv txt page file report note photo image picture screenshot scan camera iphone samsung android
""".split())

# --- (All file extraction and helper functions remain unchanged) ---
def sha256(path: Path, max_bytes: int = 300 * 1024 * 1024) -> str | None:
    try:
        if path.stat().st_size > max_bytes: return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
def sniff_text(path: Path, max_bytes=400_000) -> str:
    raw = path.read_bytes()[:max_bytes]
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try: return raw.decode(enc, errors="ignore")
    except Exception: return raw.decode("utf-8", errors="ignore")
def extract_pdf(path: Path, max_pages=5) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
    except Exception:
        return ""
def extract_docx(path: Path, max_paras=200) -> str:
    try:
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs[:max_paras])
    except Exception:
        return ""
def extract_csv_peek(path: Path, lines=120) -> str:
    try:
        out = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                out.append(line.rstrip("\n"))
                if i + 1 >= lines: break
        return "\n".join(out)
    except Exception:
        return ""
def extract_xlsx_peek(path: Path, cells=220) -> str:
    if openpyxl is None: return ""
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.active
        out, c = [], 0
        for r in ws.iter_rows(min_row=1, max_row=60, max_col=16, values_only=True):
            row = ",".join("" if v is None else str(v) for v in r)
            out.append(row); c += len(r)
            if c >= cells: break
        return "\n".join(out)
    except Exception:
        return ""
def extract_image_meta(path: Path) -> str:
    try:
        im = Image.open(str(path))
        info = {"format": im.format, "mode": im.mode, "size": im.size}
        exif_text = []
        exif = getattr(im, "getexif", lambda: None)()
        if exif:
            for k, v in exif.items():
                tag = ExifTags.TAGS.get(k, str(k))
                if isinstance(v, bytes): continue
                exif_text.append(f"{tag}: {v}")
        return json.dumps(info) + "\n" + "\n".join(exif_text[:40])
    except Exception:
        return ""
def extract_image_ocr_easyocr(reader, path: Path) -> str:
    if reader is None: return ""
    try:
        results = reader.readtext(str(path), detail=1)
        lines = [r[1] for r in results if isinstance(r, (list, tuple)) and len(r) >= 2]
        return "\n".join(lines)
    except Exception:
        return ""
def read_snippet(path: Path, mime: str, reader=None, max_bytes=400_000) -> Tuple[str, Dict[str, Any]]:
    ext = path.suffix.lower()
    meta = {"file_name": path.name, "mime": mime, "size": path.stat().st_size}
    text = ""
    if mime.startswith("text/") or ext in {".py",".js",".ts",".java",".cpp",".c",".cs",".go",".rs",".html",".css",".json",".md",".yml",".yaml"}:
        text = sniff_text(path, max_bytes=max_bytes)
    elif ext == ".pdf":
        text = extract_pdf(path)
    elif ext == ".docx":
        text = extract_docx(path)
    elif ext in {".csv", ".tsv"}:
        text = extract_csv_peek(path)
    elif ext in {".xlsx"}:
        text = extract_xlsx_peek(path)
    elif mime.startswith("image/"):
        meta_s = extract_image_meta(path)
        ocr = extract_image_ocr_easyocr(reader, path)
        text = (meta_s + ("\n\nOCR:\n" + ocr if ocr else "")).strip()
    return text[:max_bytes], meta

def choose_target(base: Path, label: str, subcategory: str) -> Path:
    if label == "Personal":
        # All personal files go into a single "Personal" folder
        return base / "Personal"
    
    # All work files go into "Work" and then a specific subcategory
    safe_cat = "Work"
    safe_sub = "".join(c for c in subcategory if c.isalnum() or c in "-_ ").strip()
    
    # If the AI fails to provide a subcategory, put it in a generic "Work" folder
    return (base / safe_cat / safe_sub) if safe_sub else (base / safe_cat)

def unique_destination(dst: Path) -> Path:
    if not dst.exists(): return dst
    stem, suf = dst.stem, dst.suffix
    for i in range(1, 9999):
        cand = dst.with_name(f"{stem}-{i}{suf}")
        if not cand.exists(): return cand
    return dst.with_name(f"{stem}-dup{int(time.time())}{suf}")

# ---- MODIFIED: Two new LLM functions for the two-step process ----

def get_primary_category(client: OpenAI, file_name: str, content: str) -> Optional[str]:
    """STEP 1: Classifies content as either 'Personal' or 'Work'."""
    system_prompt = f"""
    You are an expert file organizer. Your first task is to determine if a file is 'Personal' or 'Work'.
    'Personal' includes family photos, personal documents, hobbies, etc.
    'Work' includes professional reports, code, invoices, presentations, research, etc.

    Analyze the file's name and content preview below.
    Respond with ONLY a JSON object in the format: {{"category": "ChosenCategory"}}
    """
    user_prompt = f"Filename: {file_name}\n\nFile Content Preview:\n{content[:10000]}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False, response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        category = data.get("category")
        return category if category in PRIMARY_CATEGORIES else "Personal"
    except Exception as e:
        print(f"[ERROR] DeepSeek Step 1 (Category) failed: {e}")
        return None

def get_work_subcategory(client: OpenAI, file_name: str, content: str) -> str:
    """STEP 2: For 'Work' files, determines a specific subcategory."""
    system_prompt = f"""
    You are an expert file organizer. You have determined the following file is a 'Work' document.
    Your next task is to create a concise, one-to-three-word subcategory name for it.
    
    Base your subcategory on the document's specific content.
    Examples of good subcategories: {', '.join(WORK_SUBCATEGORIES_EXAMPLES)}

    Analyze the file's name and content preview below.
    Respond with ONLY a JSON object in the format: {{"subcategory": "Generated Subcategory Name"}}
    """
    user_prompt = f"Filename: {file_name}\n\nFile Content Preview:\n{content[:10000]}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False, response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("subcategory", "General")
    except Exception as e:
        print(f"[ERROR] DeepSeek Step 2 (Subcategory) failed: {e}")
        return "General" # Fallback subcategory

def main():
    ap = argparse.ArgumentParser(description="Focused LLM file sorter (Personal vs Work) using DeepSeek")
    ap.add_argument("source", help="Folder to sort")
    ap.add_argument("--dest", default=None, help="Destination base (default: <source>/Sorted_AI)")
    ap.add_argument("--move", action="store_true", help="Move files (default: dry-run)")
    ap.add_argument("--copy", action="store_true", help="Copy files (safe mode)")
    ap.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    ap.add_argument("--ocr_langs", default="en", help="Comma-separated OCR languages")
    ap.add_argument("--ocr_gpu", action="store_true", help="Use GPU for EasyOCR if available")
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("Error: DEEPSEEK_API_KEY environment variable not set.")
    
    print("Initializing DeepSeek client...")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        sys.exit(f"Source not found: {source}")

    dest_base = Path(args.dest).expanduser().resolve() if args.dest else (source / "Sorted_AI")
    for d in (dest_base, dest_base / "Personal", dest_base / "Work"):
        d.mkdir(parents=True, exist_ok=True)

    def prelabel(path: Path) -> str | None:
        """Only pre-label non-document files to save API calls."""
        ext = path.suffix.lower()
        if ext in {".exe", ".msi", ".pkg", ".dmg"}: return "Installers"
        if ext in {".zip",".rar",".7z",".tar",".gz",".bz2",".xz"}: return "Archives"
        if ext in {".mp3", ".wav", ".m4a", ".flac"}: return "Audio"
        if ext in {".mp4", ".mov", ".avi", ".mkv"}: return "Video"
        return None

    if easyocr is None:
        print("[WARN] easyocr not installed; OCR will be skipped.")
        reader = None
    else:
        langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()]
        print(f"Loading EasyOCR ({'+'.join(langs)}), gpu={args.ocr_gpu} …")
        reader = easyocr.Reader(langs, gpu=args.ocr_gpu)

    log_csv = dest_base / "_logs" / "moves.csv"
    new_log = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        if new_log:
            writer.writerow(["ts","action","from","to","label","subcategory","hash","model"])

        processed = 0
        for entry in source.iterdir():
            if entry.is_dir(): continue
            if args.limit and processed >= args.limit: break
            try:
                mime, _ = mimetypes.guess_type(entry.name)
                mime = mime or "application/octet-stream"
                
                label, subcat = "Other", ""
                
                # Check for non-document files first
                hardcoded_label = prelabel(entry)
                if hardcoded_label:
                    label, subcat = hardcoded_label, ""
                    # Create a folder for these types if they don't exist
                    (dest_base / label).mkdir(exist_ok=True)
                else:
                    # It's a document, so let the AI read it and decide
                    text, meta = read_snippet(entry, mime, reader=reader)
                    
                    # Step 1: Is it Personal or Work?
                    primary_category = get_primary_category(client, entry.name, text)
                    time.sleep(1) # Respect API rate limits

                    if primary_category == "Work":
                        label = "Work"
                        # Step 2: If Work, get a specific subcategory
                        subcat = get_work_subcategory(client, entry.name, text)
                        time.sleep(1)
                    elif primary_category == "Personal":
                        label = "Personal"
                        subcat = "" # No subcategories for Personal
                    else:
                        label = "Personal" # Default to Personal on failure

                target_dir = choose_target(dest_base, label, subcat)
                target_dir.mkdir(parents=True, exist_ok=True)
                dst = unique_destination(target_dir / entry.name)

                action = "DRYRUN"
                if args.copy:
                    shutil.copy2(str(entry), str(dst)); action = "COPY"
                elif args.move:
                    shutil.move(str(entry), str(dst)); action = "MOVE"

                file_hash = sha256(dst if action != "DRYRUN" else entry) or ""
                writer.writerow([int(time.time()), action, str(entry), str(dst), label, subcat, file_hash, "deepseek-chat"])
                print(f"[{action}] {entry.name} -> {target_dir.relative_to(dest_base)} (Main: {label}, Sub: {subcat or 'N/A'})")
                processed += 1

            except KeyboardInterrupt:
                print("\nInterrupted."); break
            except Exception as e:
                writer.writerow([int(time.time()), "ERROR", str(entry), "", "", "", f"{type(e).__name__}:{e}"])
                print(f"[ERROR] {entry.name}: {e}")

if __name__ == "__main__":
    main()
